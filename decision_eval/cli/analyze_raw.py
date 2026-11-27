"""CLI entrypoint: Analyze any saved benchmark_raw.jsonl and generate summary JSON/CSV + 6 SVG charts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_eval import charts, metrics, paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute full metrics & 6 SVG charts from a saved benchmark_raw.jsonl.")
    parser.add_argument("raw_jsonl", type=Path, help="Path to benchmark_raw.jsonl (e.g. ~/Downloads/benchmark_raw.jsonl).")
    parser.add_argument("--out-dir", type=Path, default=paths.REPORTS_DIR)
    parser.add_argument("--frontier-in-price", type=float, default=0.15)
    parser.add_argument("--frontier-out-price", type=float, default=0.60)
    args = parser.parse_args()

    records = [json.loads(ln) for ln in args.raw_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    models = sorted({r["model"] for r in records if r.get("model")})
    config = {"source": str(args.raw_jsonl), "models": models, "record_count": len(records)}

    summary = metrics.build_summary(
        records,
        models,
        config,
        frontier_in_price=args.frontier_in_price,
        frontier_out_price=args.frontier_out_price,
    )
    json_path, csv_path = metrics.write_summary_files(summary, args.out_dir)
    created = charts.generate_all_charts(summary, args.out_dir / "charts")

    print(f"Analyzed {len(records)} records across models={models}")
    print(f"  Summary JSON : {json_path}")
    print(f"  Summary CSV  : {csv_path}")
    print(f"  Charts ({len(created)})   : {args.out_dir / 'charts' / 'index.html'}")
    for m, md in summary["models"].items():
        ov = md["overall"]
        print(
            f"  {m:16s} | n={ov.get('n',0):3d} | acc={ov.get('accuracy',0)*100:5.1f}% "
            f"| ECE={ov.get('ece',0):.3f} | AURC={ov.get('aurc',0):.3f} "
            f"| parse={ov.get('parse_ok_rate',0)*100:5.1f}% | p50={ov.get('latency_p50_ms',0):6.1f}ms"
        )


if __name__ == "__main__":
    main()
