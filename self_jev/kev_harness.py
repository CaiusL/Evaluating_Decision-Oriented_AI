"""Ask Kev which built racing car to publish, then upload that zip.

Kev scores every lap-time lever. Creating a new judge runs the upload step.
The same judge does not upload again. A built car is uploaded only when each
lever it changes is the option Kev chose at probability 0.70 or higher, and no
other lever Kev chose at that bar disagrees with the car. The published car is
not uploaded again. A probability is not a lap time.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from self_jev.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "kev_harness.json"
JUDGES = ROOT / "reports" / "kev_judges"
WORKFLOW = ROOT / "state" / "workflow.json"
PUBLISH_BAR = 0.70

STATE = """Account J races one formula car. Rank 0 is first place. The best verified finish is second place, rank 1, version 5 race 1a8c2aea. No finished race has rank 0. The published car is sj7-gap. sj8-energy is built and not published.

A fair lap-time pair is two still-racing finishes, same track and same wet band, one decision different, both lap times measured. A damage retirement is not a lap time. An opponent finish is not a setup label. A comparison that also changes another knob does not name which change was faster. A probability is not a lap time.

Published sj7-gap values, the same as the finished a2585d41 setup except the stopping rule: grip speed table, preview 0.18, yaw_gain 0.7, qual_margin 1, race_margin 1, latency 0.05, ld_min 7, ld_time 0.35, kv 1.9, brake_gain 1.04, fuel 9 kg, ballast 20 kg, wings 20 front and 22 rear, radiator 0.7, tyre pressure 29 front and 28 rear psi, start compound soft, ers_harvest_level 5. One planned stop: lap 1, medium, request 180 m ahead. The energy budget is inactive until lap 1. The stopping rule eases the emergency stop when the gap ahead is stable or growing, and keeps it when the gap is closing. Launch mode holds the grid lane for lap 0 until 330 m. Passing is considered from lap 1, for a car 35 to 280 m ahead, only when the advantage exceeds 1.5.

Base finish a2585d41, sj4-pace, damp Parkland, peak wetness 0.249972, still racing, 4 laps in 445.223 s. The emergency stop is kept. Mean 52.10 m/s, peak 92.84, lateral 3.23 g. While racing the free envelope averages 76.65 m/s, the follow cap 71.42, the stopping cap 63.17, and the car 52.09. Stopping-cap cuts: 2497 meter-seconds in lane, 1118 already 2.7 m off line. Lap 0 is soft: J 110.033 s versus the winner 92.917 s, and J is 3.03 s behind at 400 m. Grid order is 9, 2, 3, 0. Qualifying is 91.026 s versus 89.641 s. At lights the envelope is 90 m/s and the guard target is 5.3 m/s. Seat 2 is 16 m ahead and 0.24 m off line. Recorded throttle is 0.43. Replay throttle is 0.31 and brake is 0. Cars with nobody in their lane show throttle 1. Clean medium lap 3, tyres 57 to 63 C, straights within 6 m/s of the grip envelope. Qualifying in clear air is within 2 m/s of the envelope from 0 to 1000 m. At s 742 on lap 3 the car is at 82 m/s with brake 0.21, while the replay brake is 0 and the guard cap is 81.7. Seat 0 race ERS mode 3 occurs 114 times and state of charge falls to 0.03 MJ.

Version 6 sj7-gap has no still-racing damp Parkland finish. The checked results are damage retirements: 21f76602 Highland on lap 2, c767b15a wet Parkland on lap 0, ebe0e4b9 damp Parkland on lap 3, 049a8010 Highland after 1 lap, f96edeef Highland after 3 laps, 37b81ebd Highland on lap 0, and 08569f9b after 2 laps. The latest finished match is ede97a2e at 2026-11-19T06:16:21Z, seat 8, classification rank 1, on crossover. The track length is 5803.17 m. Weather went dry, then damp, then wet, and the peak wetness is 0.999972. The public design is fuel 9 kg, ballast 20 kg, wings 20 front and 22 rear, radiator 0.7, ers harvest 5, and soft, the same book as the published car. Qualifying is valid at 137.931 s. Grid order is 1, 8, 9, 4, 2, 0, 5, 6, 3, 7. The lap-1 event is 127.717 s and the lap-2 event is 120.906 s. J has no pit-exit event. J retired from damage on lap 3. The race closed because the absolute time limit was reached. J has no finish event. Crossover is not damp Parkland, and a damage retirement is not a still-racing lap time.

Wet Parkland 5f2ea146, version 4, still racing, rank 3, 4 laps in 458.811 s. Peak wetness 0.749972, so it is not the same band as a2585d41. J did not pit. J lap 0 is 126.83 s. The battery is 3.0 MJ at lights, 0.49 MJ by 673 m, and 0.05 MJ by 1276 m, before the later energy windows. The budget stays off until lap 1. Seats that pitted to medium and back to soft finished behind J. That is not a pit label.

Damp Parkland speed comparison: the cold car 6214cc20 averages 53.64 m/s and peaks at 95.63. The grip car a2585d41 averages 52.10 and peaks at 92.84. Those races also changed preview from 0.18 to 0.33, yaw gain from 0.7 to 0.8, fuel from 9 to 11 kg, and ballast from 20 to 40 kg. Every kept outcome pair has race_margin 1. The only other played value is 0.995, and those logs do not record wetness, so they are not a pair. Wings, radiator, pressures, start compound, harvest level, qual_margin, kv, latency, and brake_gain have no isolated still-racing pair."""

# Values the published car already races. A built car lists only the levers it changes.
HELD = {
    "speed_table": "stay_grip",
    "launch": "keep_launch",
    "passing": "keep_pass",
    "pit": "keep_stop",
    "preview": "keep",
    "yaw": "keep",
    "race_margin": "keep",
    "fuel": "keep",
    "ballast": "keep",
    "wings": "keep",
    "brake": "keep",
    "straight_cap": "keep",
    "ers_harvest": "keep",
    "cooling_tyres": "keep",
    "filters": "keep",
    "stopping": "stay_eased",
    "energy": "leave_off",
}

QUESTIONS = {
    "any_fair_pair": {
        "type": "noul",
        "instructions": "Does any valuable lap-time lever already have a fair pair? A fair pair is two still-racing finishes, same track and wet band, one decision different, both lap times measured. A damage retirement is not a lap time. A comparison that changes more than one knob is not a fair pair.",
    },
    "stopping": {
        "type": "choice",
        "instructions": "What should the next published car do about the stopping rule?",
        "criteria": {
            "stay_eased": "Keep sj7-gap: ease the stop when the gap ahead is stable or growing, and keep the emergency stop when the gap is closing.",
            "restore_stop": "Restore the emergency stop for every car ahead, as on the finished a2585d41 car.",
            "unresolved": "No still-racing pair isolates this rule, so do not change it.",
        },
    },
    "energy": {
        "type": "choice",
        "instructions": "What should the next published car do about the lap-0 energy budget?",
        "criteria": {
            "leave_off": "Leave the budget inactive until lap 1, as on the published car.",
            "turn_on": "Turn the budget on from lap 0 so the battery is not spent before the later windows.",
            "unresolved": "No still-racing pair isolates lap-0 deployment, so do not change it.",
        },
    },
    "speed_table": {
        "type": "choice",
        "instructions": "What should the next published car do about grip versus the cold speed table?",
        "criteria": {
            "stay_grip": "Keep the grip table.",
            "switch_cold": "Switch to the cold table.",
            "unresolved": "The damp Parkland races also changed preview, yaw, fuel, and ballast, so they do not name the faster table.",
        },
    },
    "launch": {
        "type": "choice",
        "instructions": "What should the next published car do about the launch, lap 0 until 330 m?",
        "criteria": {
            "keep_launch": "Keep grid-launch: hold the lane, and use the low throttle the replay shows when a car is ahead in the lane.",
            "open_launch": "Use full throttle at lights when the car's own lane is clear.",
            "unresolved": "No still-racing pair isolates the launch rule, so do not change it.",
        },
    },
    "passing": {
        "type": "choice",
        "instructions": "What should the next published car do about passing?",
        "criteria": {
            "keep_pass": "Keep the lap-1 pass: a car 35 to 280 m ahead, and only when the advantage exceeds 1.5.",
            "pass_sooner": "Allow a pass on lap 0 as well.",
            "unresolved": "No still-racing pair isolates the pass rule, so do not change it.",
        },
    },
    "pit": {
        "type": "choice",
        "instructions": "What should the next published car do about the planned lap-1 stop from soft to medium?",
        "criteria": {
            "keep_stop": "Keep the lap-1 medium stop.",
            "skip_stop": "Do not plan that stop.",
            "unresolved": "No still-racing pair isolates the stop, so do not change it.",
        },
    },
    "preview": {
        "type": "choice",
        "instructions": "What should the next published car do about preview, now 0.18?",
        "criteria": {
            "keep": "Keep preview 0.18.",
            "raise": "Raise preview to 0.33.",
            "unresolved": "Preview never changes alone, so do not change it.",
        },
    },
    "yaw": {
        "type": "choice",
        "instructions": "What should the next published car do about yaw gain, now 0.7?",
        "criteria": {
            "keep": "Keep yaw gain 0.7.",
            "raise": "Raise yaw gain to 0.8.",
            "unresolved": "Yaw gain never changes alone, so do not change it.",
        },
    },
    "race_margin": {
        "type": "choice",
        "instructions": "What should the next published car do about race_margin, now 1?",
        "criteria": {
            "keep": "Keep race_margin 1.",
            "cut": "Cut race_margin to 0.995.",
            "unresolved": "The 0.995 races have no recorded wetness, so do not change it.",
        },
    },
    "fuel": {
        "type": "choice",
        "instructions": "What should the next published car do about fuel, now 9 kg?",
        "criteria": {
            "keep": "Keep 9 kg.",
            "add": "Raise fuel to 11 kg.",
            "unresolved": "Fuel never changes alone, so do not change it.",
        },
    },
    "ballast": {
        "type": "choice",
        "instructions": "What should the next published car do about ballast, now 20 kg?",
        "criteria": {
            "keep": "Keep 20 kg.",
            "add": "Raise ballast to 40 kg.",
            "unresolved": "Ballast never changes alone, so do not change it.",
        },
    },
    "wings": {
        "type": "choice",
        "instructions": "What should the next published car do about the wings, now 20 front and 22 rear?",
        "criteria": {
            "keep": "Keep 20/22.",
            "change": "Publish a different wing pair and nothing else.",
            "unresolved": "No still-racing pair isolates the wings, so do not change them.",
        },
    },
    "brake": {
        "type": "choice",
        "instructions": "What should the next published car do about brake gain, now 1.04? One damp frame shows recorded brake 0.21 where the replay brake is 0 under the stopping cap.",
        "criteria": {
            "keep": "Keep brake gain 1.04.",
            "change": "Publish a different brake gain and nothing else.",
            "unresolved": "That frame is the stopping guard, not an isolated brake-gain pair, so do not change it.",
        },
    },
    "straight_cap": {
        "type": "choice",
        "instructions": "What should the next published car do about the straight speed cap? Measured straights are within 6 m/s of the grip envelope, and the written notes say those gaps are not a higher-cap label.",
        "criteria": {
            "keep": "Keep the grip floor.",
            "raise": "Raise the straight cap.",
            "unresolved": "No still-racing pair isolates a higher cap, so do not change it.",
        },
    },
    "ers_harvest": {
        "type": "choice",
        "instructions": "What should the next published car do about ERS harvest level, now 5?",
        "criteria": {
            "keep": "Keep harvest level 5.",
            "change": "Publish a different harvest level and nothing else.",
            "unresolved": "No still-racing pair isolates harvest level, so do not change it.",
        },
    },
    "cooling_tyres": {
        "type": "choice",
        "instructions": "What should the next published car do about radiator 0.7, tyre pressure 29/28, and the soft start compound?",
        "criteria": {
            "keep": "Keep radiator, pressures, and soft.",
            "change": "Publish a different cooling, pressure, or compound and nothing else.",
            "unresolved": "No still-racing pair isolates these, so do not change them.",
        },
    },
    "filters": {
        "type": "choice",
        "instructions": "What should the next published car do about qual_margin 1, latency 0.05, ld_min 7, ld_time 0.35, and kv 1.9?",
        "criteria": {
            "keep": "Keep these filter values.",
            "change": "Publish a different value of one of them and nothing else.",
            "unresolved": "No still-racing pair isolates these, so do not change them.",
        },
    },
}

# sj8-energy turns the lap-0 budget on and restores the emergency stop. sj4-pace is that stop without the budget.
CARS = {
    "sj7-gap": {
        "archive": ROOT / "submissions" / "racing" / "sj7-gap.zip",
        "choices": {"stopping": "stay_eased", "energy": "leave_off"},
    },
    "sj4-pace": {
        "archive": ROOT / "submissions" / "racing" / "sj4-pace.zip",
        "choices": {"stopping": "restore_stop", "energy": "leave_off"},
    },
    "sj8-energy": {
        "archive": ROOT / "submissions" / "racing" / "sj8-energy.zip",
        "choices": {"stopping": "restore_stop", "energy": "turn_on"},
    },
}
INCUMBENT = "sj7-gap"


def kev_python() -> Path:
    chosen = os.environ.get("KEV_PYTHON")
    if chosen:
        return Path(chosen)
    home = Path(os.environ.get("KEV_HOME", "/tmp/kev-src"))
    return home / ".venv" / "bin" / "python"


def ask_kev(state: str, questions: dict) -> dict:
    """Run jaredpalmer/kev-0.8b in the Kev interpreter. stdout is the answer object."""
    python = kev_python()
    if not python.is_file():
        raise RuntimeError(f"Kev interpreter not found at {python}. Set KEV_PYTHON.")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    handle, out_path = tempfile.mkstemp(prefix="kev-ask-", suffix=".json")
    os.close(handle)
    proc = subprocess.run(
        [str(python), "-m", "self_jev.kev_ask", out_path],
        input=json.dumps({"state": state, "questions": questions}).encode(),
        stdout=subprocess.PIPE,
        cwd=ROOT,
        env=env,
        timeout=600,
        check=False,
    )
    written = Path(out_path)
    try:
        if proc.returncode != 0 or written.stat().st_size == 0:
            detail = (proc.stdout or b"").decode()[-2000:]
            raise RuntimeError(f"Kev failed ({proc.returncode}): {detail}")
        return json.loads(written.read_text())
    finally:
        written.unlink(missing_ok=True)


def _choice(answer: dict) -> tuple[str, float]:
    picked = answer["choice"]
    return picked, float(answer["probabilities"][picked])


def select_car(answers: dict, cars: dict, held: dict, incumbent: str, bar: float = PUBLISH_BAR) -> dict | None:
    """The one built car Kev's bar-clearing choices agree with, excluding the published car."""
    demanded = {}
    for lever, answer in answers.items():
        if answer.get("type") != "choice":
            continue
        picked, probability = _choice(answer)
        if picked != "unresolved" and probability >= bar:
            demanded[lever] = picked
    winners = []
    published = cars[incumbent]["choices"]
    for name, car in cars.items():
        full = {**held, **car["choices"]}
        if any(full.get(lever) != picked for lever, picked in demanded.items()):
            continue
        changed = {
            lever: picked
            for lever, picked in car["choices"].items()
            if picked != published.get(lever, held.get(lever))
        }
        if not changed:
            continue
        if any(
            lever not in answers
            or answers[lever].get("choice") != picked
            or _choice(answers[lever])[1] < bar
            for lever, picked in changed.items()
        ):
            continue
        winners.append(name)
    if len(winners) != 1:
        return None
    name = winners[0]
    return {"name": name, "archive": str(cars[name]["archive"]), "choices": cars[name]["choices"]}


def _incumbent() -> dict:
    state = json.loads(WORKFLOW.read_text())
    return state["games"]["racing"]["incumbent"]


def judge_key(report: dict) -> str:
    """Identity of a judgment. Latency and the receipt are not part of it."""
    selected = report.get("selected")
    body = {
        "model": report.get("model"),
        "bar": report.get("bar"),
        "incumbent": report.get("incumbent"),
        "state_sha256": report.get("state_sha256"),
        "answers": report.get("answers"),
        "selected": None if not selected else {"name": selected["name"], "choices": selected["choices"]},
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {key: value for key, value in body.items() if key != "new"}
    path.write_text(json.dumps(stored, indent=2) + "\n")


def _store_judge(report: dict, judges_dir: Path, report_path: Path) -> None:
    path = judges_dir / f"{report['id']}.json"
    _write_json(path, report)
    _write_json(judges_dir / "latest.json", {"id": report["id"], "path": str(path)})
    _write_json(report_path, report)


def upload_judge(report: dict, *, submit=None, remember: bool = False) -> dict:
    """Upload step. A judge runs this once, when it is first created."""
    if report.get("upload_step") == "ran":
        return report
    report["upload_step"] = "ran"
    report["uploaded"] = False
    report["receipt"] = None
    chosen = report.get("selected")
    if not chosen:
        return report
    archive = Path(chosen["archive"])
    receipt = (submit or _submit)("racing", archive)
    report["uploaded"] = True
    report["receipt"] = receipt
    if remember:
        _remember(chosen["name"], archive, receipt)
    return report


def upload_latest(*, submit=None, remember: bool = True, judges_dir: Path | None = None,
                  report_path: Path | None = None) -> dict:
    """Run the upload step for the latest judge when that step is still held."""
    directory = JUDGES if judges_dir is None else judges_dir
    pointer_path = directory / "latest.json"
    if not pointer_path.is_file():
        raise RuntimeError("no judge has been created")
    pointer = json.loads(pointer_path.read_text())
    path = Path(pointer["path"])
    report = json.loads(path.read_text())
    if report.get("upload_step") == "ran":
        report["new"] = False
        return report
    report = upload_judge(report, submit=submit, remember=remember)
    _store_judge(report, directory, REPORT if report_path is None else report_path)
    report["new"] = False
    return report


def _remember(name: str, archive: Path, receipt: dict) -> None:
    from self_jev.workflow import load, save

    state = load()
    state["games"]["racing"]["incumbent"] = {
        "archive": str(archive),
        "sha256": receipt["sha256"],
        "submission_id": receipt.get("submission_id"),
        "version_id": receipt.get("version_id"),
        "name": name,
    }
    save(state)


def run(*, upload: bool = True, ask=None, submit=None, report_path: Path | None = None,
        cars: dict | None = None, held: dict | None = None, incumbent: dict | None = None,
        bar: float = PUBLISH_BAR, remember: bool = False, judges_dir: Path | None = None) -> dict:
    """Create a judge. A new judge runs the upload step."""
    cars = CARS if cars is None else cars
    held = HELD if held is None else held
    incumbent = _incumbent() if incumbent is None else incumbent
    directory = JUDGES if judges_dir is None else judges_dir
    destination = REPORT if report_path is None else report_path
    payload = (ask or ask_kev)(STATE, QUESTIONS)
    if isinstance(payload, dict) and isinstance(payload.get("answers"), dict):
        answers = payload["answers"]
        meta = {
            key: payload[key]
            for key in ("backend", "dtype", "temperature", "input_tokens", "state_tokens", "state_truncated", "latency_ms", "base_source")
            if key in payload
        }
    else:
        answers = payload
        meta = {}
    chosen = select_car(answers, cars, held, incumbent["name"], bar)
    if chosen and sha256_file(Path(chosen["archive"])) == incumbent.get("sha256"):
        chosen = None
    report = {
        "model": "jaredpalmer/kev-0.8b",
        "bar": bar,
        "incumbent": incumbent.get("name"),
        "state_sha256": hashlib.sha256(STATE.encode()).hexdigest()[:16],
        **meta,
        "answers": answers,
        "selected": chosen,
        "uploaded": False,
        "receipt": None,
    }
    report["id"] = judge_key(report)
    path = directory / f"{report['id']}.json"
    if path.is_file():
        stored = json.loads(path.read_text())
        _write_json(destination, stored)
        stored["new"] = False
        return stored
    if upload:
        report = upload_judge(report, submit=submit, remember=remember)
    else:
        report["upload_step"] = "held"
    _store_judge(report, directory, destination)
    report["new"] = True
    return report


def _submit(game: str, archive: Path) -> dict:
    from self_jev.submit import submit_zip

    return submit_zip(game, archive, select=True)


def print_report(report: dict) -> None:
    created = "new judge" if report.get("new") else "judge already created"
    print(f"{created} {report.get('id')}")
    print(f"report {REPORT}")
    print(f"upload step {report.get('upload_step')}")
    selected = report.get("selected")
    if report.get("uploaded"):
        version = (report.get("receipt") or {}).get("version_id")
        print(f"uploaded {selected['name']} version {version}")
        return
    if selected and report.get("upload_step") == "held":
        print(f"would upload {selected['name']}. Left unpublished.")
        return
    print(f"Kev judged {len(QUESTIONS)} questions. No built car cleared {PUBLISH_BAR:.2f}. Nothing uploaded.")


def main() -> None:
    print_report(run(upload=True, remember=True))
