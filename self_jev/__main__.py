"""Run Self-JEV.

    python -m self_jev status
    python -m self_jev observe
    python -m self_jev check
    python -m self_jev first
    python -m self_jev draft
    python -m self_jev train-elements
    python -m self_jev draft-racing
    python -m self_jev train-outcome
    python -m self_jev train-pace-outcome
    python -m self_jev draft-outcome
    python -m self_jev cycle
    python -m self_jev generate
    python -m self_jev train
    python -m self_jev benchmark
    python -m self_jev capture
    python -m self_jev submit --game racing
    python -m self_jev submit --game racing --archive submissions/racing/sj6-track.zip
    python -m self_jev kev
    python -m self_jev kev --no-submit
    python -m self_jev kev-upload
    python -m self_jev kev-ui
"""
from __future__ import annotations

import argparse
from pathlib import Path

from self_jev.check import check_bundle
from self_jev.workflow import cycle, load, observe, status_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-JEV RSI workflow")
    parser.add_argument("command", choices=("status", "observe", "check", "first", "draft", "draft-racing", "train-elements", "train-racing", "train-outcome", "train-pace-outcome", "draft-outcome", "cycle", "generate", "train", "benchmark", "capture", "submit", "kev", "kev-upload", "kev-ui"))
    parser.add_argument("--game", choices=("fps", "racing"))
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--no-select", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        from self_jev.generate import main as generate_main
        generate_main()
        return
    if args.command == "train":
        from self_jev.train import main as train_main
        train_main()
        return
    if args.command == "benchmark":
        from self_jev.benchmark import main as benchmark_main
        benchmark_main()
        return
    if args.command == "capture":
        from self_jev.capture import main as capture_main
        capture_main()
        return
    if args.command == "submit":
        from self_jev.submit import run
        run(args.game or "racing", args.archive, args.directory, select=not args.no_select)
        return
    if args.command == "kev":
        from self_jev.kev_harness import print_report, run as kev_run
        print_report(kev_run(upload=not args.no_submit, remember=not args.no_submit))
        return
    if args.command == "kev-upload":
        from self_jev.kev_harness import print_report, upload_latest
        print_report(upload_latest(remember=True))
        return
    if args.command == "kev-ui":
        from self_jev.kev_tui import main as kev_ui_main
        kev_ui_main()
        return
    if args.command == "first":
        from self_jev.first import main as first_main
        first_main()
        return
    if args.command == "draft":
        from self_jev.elements import draft_main
        draft_main()
        return
    if args.command == "train-elements":
        from self_jev.element_train import main as element_train_main
        element_train_main()
        return
    if args.command == "train-racing":
        from self_jev.racing_pace import train as train_racing
        train_racing()
        return
    if args.command == "draft-racing":
        from self_jev.racing_pace import draft_racing
        draft_racing()
        return
    if args.command == "train-outcome":
        from self_jev.outcome_train import main as train_outcome_main
        train_outcome_main()
        return
    if args.command == "train-pace-outcome":
        from self_jev.pace_outcome import main as train_pace_outcome_main
        train_pace_outcome_main()
        return
    if args.command == "draft-outcome":
        from self_jev.outcome_draft import draft_outcome
        draft_outcome()
        return
    if args.command == "status":
        print("\n".join(status_lines()))
        return
    if args.command == "observe":
        state = observe()
        for game, row in state["games"].items():
            matches = row["latest_own_matches"]
            print(f"{game}: {len(matches)} own finished match(es)")
            for item in matches[:5]:
                print(f"  {item['finished_at']} {item['match_id']} worse={item['worse']} rank={item['rank']}")
        return
    if args.command == "check":
        names = [args.game] if args.game else ["fps", "racing"]
        state = load()
        for game in names:
            archive = Path(state["games"][game]["incumbent"]["archive"])
            print(f"{game}: {check_bundle(archive, game)}")
        return
    print("\n".join(cycle(args.game, submit=not args.no_submit)))


if __name__ == "__main__":
    main()
