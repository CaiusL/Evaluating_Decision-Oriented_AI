#!/usr/bin/env python3
"""Validate and merge teacher-generated JSONL batches."""

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = [
    "id",
    "source_id",
    "bucket",
    "context_band",
    "context",
    "question",
    "options",
    "option_descriptions",
    "gold_index",
    "gold_label",
    "abstain_index",
    "is_abstain_gold",
    "rule_proof",
]


def validate_rows(paths: list[Path], bench_path: Path | None = None) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errs: list[str] = []
    seen_ids: set[str] = set()
    seen_sids: set[str] = set()
    seen_ctxs: set[str] = set()

    bench_ctxs: set[str] = set()
    if bench_path and bench_path.is_file():
        for line in bench_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                bench_ctxs.add(json.loads(line)["context"])

    for p in paths:
        for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                errs.append(f"{p.name}:{idx}: invalid JSON ({e})")
                continue

            rid = str(r.get("id", f"<line {idx}>"))
            missing = [k for k in REQUIRED_KEYS if k not in r]
            if missing:
                errs.append(f"{p.name}:{idx} ({rid}): missing keys {missing}")
                continue

            opts = r["options"]
            descs = r["option_descriptions"]
            gi = r["gold_index"]
            ai = r["abstain_index"]
            if len(opts) != len(descs):
                errs.append(f"{p.name}:{idx} ({rid}): len(options) != len(option_descriptions)")
            if not (0 <= gi < len(opts)):
                errs.append(f"{p.name}:{idx} ({rid}): gold_index {gi} out of range")
            elif opts[gi] != r["gold_label"]:
                errs.append(f"{p.name}:{idx} ({rid}): gold_label mismatch")
            if r["is_abstain_gold"] != (gi == ai):
                errs.append(f"{p.name}:{idx} ({rid}): is_abstain_gold mismatch")

            if rid in seen_ids:
                errs.append(f"{p.name}:{idx} ({rid}): duplicate id")
            if r["source_id"] in seen_sids:
                errs.append(f"{p.name}:{idx} ({rid}): duplicate source_id {r['source_id']}")
            if r["context"] in seen_ctxs:
                errs.append(f"{p.name}:{idx} ({rid}): duplicate context")
            if r["context"] in bench_ctxs:
                errs.append(f"{p.name}:{idx} ({rid}): overlaps all_test.jsonl")

            seen_ids.add(rid)
            seen_sids.add(r["source_id"])
            seen_ctxs.add(r["context"])
            rows.append(r)

    return rows, errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--bench",
        type=Path,
        default=Path("/Users/caius/Projects/self_jev/data/benchmarks/all_test.jsonl"),
    )
    args = parser.parse_args()

    rows, errs = validate_rows(args.inputs, args.bench)
    print(f"Valid rows: {len(rows)}, Errors: {len(errs)}")
    for e in errs[:20]:
        print("  ERR:", e)

    print("Buckets:", dict(Counter(r["bucket"] for r in rows)))
    print("Context bands:", dict(Counter(r["context_band"] for r in rows)))
    print("Abstain:", dict(Counter(r["is_abstain_gold"] for r in rows)))
    print("Gold indices:", dict(sorted(Counter(r["gold_index"] for r in rows).items())))

    if args.out and not errs:
        args.out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        print(f"Merged {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
