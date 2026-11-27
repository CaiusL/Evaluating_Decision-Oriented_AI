"""Full-screen terminal view of the Kev harness.

    python -m self_jev kev-ui

j creates a judge and holds the upload. u runs the upload step. q leaves the screen.
A pipe with no terminal prints the screen once.
"""
from __future__ import annotations

import json
import sys
import termios
import tty
from pathlib import Path

from self_jev.kev_harness import CARS, JUDGES, PUBLISH_BAR, QUESTIONS


def load_latest(judges_dir: Path | None = None) -> dict | None:
    directory = JUDGES if judges_dir is None else judges_dir
    pointer = directory / "latest.json"
    if not pointer.is_file():
        return None
    path = Path(json.loads(pointer.read_text())["path"])
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _mark(probability: float, bar: float, actionable: bool) -> str:
    return "*" if actionable and probability >= bar else " "


def _answer_line(lever: str, answer: dict, bar: float) -> str:
    if answer.get("type") == "noul":
        probability = float(answer["noul"])
        return f"{_mark(probability, bar, True)} {lever:<16} p(yes) {probability:.4f}"
    picked = answer["choice"]
    probabilities = answer["probabilities"]
    probability = float(probabilities[picked])
    others = sorted(
        ((name, float(value)) for name, value in probabilities.items() if name != picked),
        key=lambda item: item[1],
        reverse=True,
    )
    nxt = f"   next {others[0][0]} {others[0][1]:.4f}" if others else ""
    return f"{_mark(probability, bar, picked != 'unresolved')} {lever:<16} {picked:<14} {probability:.4f}{nxt}"


def _verdict(report: dict | None) -> str:
    if not report:
        return "no judge yet"
    selected = report.get("selected")
    if report.get("uploaded") and selected:
        version = (report.get("receipt") or {}).get("version_id")
        return f"uploaded {selected['name']} version {version}"
    if selected and report.get("upload_step") == "held":
        return f"would upload {selected['name']}. upload step held"
    if report.get("upload_step") == "ran":
        return "upload step ran. no built car cleared the bar"
    return "upload step has not run"


def screen(report: dict | None, status: str = "") -> str:
    """The plain text of one screen. The interactive loop clears the terminal around it."""
    bar = float(report["bar"]) if report and report.get("bar") is not None else PUBLISH_BAR
    lines = [
        "Kev harness",
        f"bar {bar:.2f}   incumbent {report.get('incumbent') if report else '-'}",
        f"judge {report.get('id') if report else '-'}   {_verdict(report)}",
    ]
    if report and report.get("backend"):
        lines.append(
            f"{report.get('backend')}  {report.get('dtype') or ''}  "
            f"{report.get('input_tokens')} tokens  {report.get('latency_ms')} ms"
        )
    lines.append("")
    lines.append(f"{'':1} {'lever':<16} {'call':<14} p")
    answers = (report or {}).get("answers") or {}
    order = [key for key in QUESTIONS if key in answers] or list(answers)
    if not order:
        lines.append("  no answers yet")
    for lever in order:
        lines.append(_answer_line(lever, answers[lever], bar))
    lines.append("")
    lines.append("built cars")
    for name, car in CARS.items():
        bits = "  ".join(f"{lever}={choice}" for lever, choice in car["choices"].items())
        lines.append(f"  {name:<12} {bits}")
    lines.append("")
    lines.append("* at or above the bar")
    lines.append("j  create a judge and hold the upload")
    lines.append("u  run the upload step")
    lines.append("q  quit")
    if status:
        lines.append("")
        lines.append(status)
    return "\n".join(lines)


def _upload_status(report: dict) -> str:
    selected = report.get("selected")
    if report.get("uploaded") and selected:
        version = (report.get("receipt") or {}).get("version_id")
        return f"uploaded {selected['name']} version {version}"
    if report.get("upload_step") == "held" and selected:
        return f"would upload {selected['name']}. upload step held"
    return "upload step finished. nothing uploaded."


def serve(keys, *, judge, upload, write, judges_dir: Path | None = None) -> dict | None:
    """Draw the screen, then apply each key. j judges. u uploads. q returns."""
    report = load_latest(judges_dir)
    status = ""
    incoming = iter(keys)
    while True:
        write(screen(report, status))
        try:
            key = next(incoming)
        except StopIteration:
            return report
        if key == "q":
            return report
        if key == "j":
            write("scoring with Kev")
            try:
                report = judge(judges_dir=judges_dir)
                status = "new judge created. upload step held." if report.get("new") else "judge already created."
            except RuntimeError as exc:
                status = str(exc)
            continue
        if key == "u":
            try:
                report = upload(judges_dir=judges_dir)
                status = _upload_status(report)
            except RuntimeError as exc:
                status = str(exc)
            continue
        status = "keys: j judge, u upload, q quit"


def _read_key() -> str:
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    if char == "\x03":
        raise KeyboardInterrupt
    return char


def _keys():
    while True:
        yield _read_key()


def _draw(text: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\033[H\033[2J")
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def main() -> None:
    from self_jev.kev_harness import run, upload_latest

    if not sys.stdin.isatty():
        print(screen(load_latest()))
        return

    def judge(judges_dir=None):
        return run(upload=False, remember=False, judges_dir=judges_dir)

    def upload(judges_dir=None):
        return upload_latest(remember=True, judges_dir=judges_dir)

    serve(_keys(), judge=judge, upload=upload, write=_draw)
