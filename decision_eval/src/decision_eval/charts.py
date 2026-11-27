"""Generate publication-ready SVG charts from reports/full_eval/eval_summary.json.

Produces 6 clean vector charts (pure Python stdlib, zero matplotlib/numpy required):
  1. 01_cascade_pareto.svg      - Accuracy vs. Cost per 1,000 Decisions (Confidence-Gated Cascade Pareto)
  2. 02_reliability_diagram.svg - Calibration Reliability Diagram (Confidence vs. Observed Accuracy)
  3. 03_risk_coverage.svg       - Selective Prediction Risk-Coverage Curve (AURC)
  4. 04_suite_accuracy.svg      - Accuracy Comparison Across All 4 Benchmark Suites
  5. 05_context_length.svg      - Context-Length Degradation Curve (Short / Medium / Long)
  6. 06_latency_comparison.svg  - Inference Latency Comparison (p50 & p95 ms)

Also writes index.html in the charts directory so all 6 charts can be viewed side-by-side in any browser.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from decision_eval.paths import CHARTS_DIR as DEFAULT_OUT
from decision_eval.paths import REPORTS_DIR

DEFAULT_SUMMARY = REPORTS_DIR / "benchmark_summary.json"

PALETTE = {
    "kev-0.8b": "#2563eb",       # Blue
    "qwen-0.8b-ll": "#d97706",   # Amber
    "qwen-0.8b-json": "#dc2626", # Red
    "frontier-llm": "#059669",   # Emerald
    "self-jev-minilm": "#0891b2",# Teal
    "nn-combo": "#7c3aed",       # Purple (car-build-nn + jev-rl)
    "car-build-nn": "#7c3aed",   # Purple (legacy alias)
    "kev-0.8b->frontier-llm": "#2563eb",
    "qwen-0.8b-ll->frontier-llm": "#d97706",
    "qwen-0.8b-json->frontier-llm": "#dc2626",
}


def _color(name: str, fallback_idx: int = 0) -> str:
    defaults = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    return PALETTE.get(name, defaults[fallback_idx % len(defaults)])


def _svg_header(w: int, h: int, title: str, subtitle: str = "") -> list[str]:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;fill:#0f172a;} '
        '.sub{fill:#475569;font-size:12px;} .axis{stroke:#94a3b8;stroke-width:1.2;} '
        '.grid{stroke:#e2e8f0;stroke-width:1;stroke-dasharray:3,3;} .tick{fill:#475569;font-size:11px;} '
        '.lbl{fill:#1e293b;font-size:12px;font-weight:600;}</style>',
        f'<rect width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="#e2e8f0"/>',
        f'<text x="24" y="30" font-size="16" font-weight="700">{html.escape(title)}</text>',
    ]
    if subtitle:
        out.append(f'<text x="24" y="48" class="sub">{html.escape(subtitle)}</text>')
    return out


def _legend(items: list[tuple[str, str]], x0: int, y0: int) -> list[str]:
    out = []
    x = x0
    for label, col in items:
        out.append(f'<rect x="{x}" y="{y0-10}" width="12" height="12" rx="2" fill="{col}"/>')
        out.append(f'<text x="{x+16}" y="{y0}" font-size="11" fill="#334155">{html.escape(label)}</text>')
        x += max(95, len(label) * 7 + 28)
    return out


def _plot_line_chart(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    diagonal: bool = False,
    w: int = 760,
    h: int = 440,
) -> str:
    pl, pr, pt, pb = 72, 32, 86, 62
    pw, ph = w - pl - pr, h - pt - pb

    def sx(v: float) -> float:
        return pl + ((v - x_min) / max(1e-9, x_max - x_min)) * pw

    def sy(v: float) -> float:
        return pt + ph - ((v - y_min) / max(1e-9, y_max - y_min)) * ph

    out = _svg_header(w, h, title, subtitle)
    leg = [(name, _color(name, idx)) for idx, name in enumerate(series.keys())]
    out.extend(_legend(leg, 24, 70))

    # Grid + ticks
    for i in range(6):
        yv = y_min + (y_max - y_min) * (i / 5.0)
        yy = sy(yv)
        out.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+pw}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{pl-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{yv:.2f}</text>')
        xv = x_min + (x_max - x_min) * (i / 5.0)
        xx = sx(xv)
        out.append(f'<line x1="{xx:.1f}" y1="{pt}" x2="{xx:.1f}" y2="{pt+ph}" class="grid"/>')
        out.append(f'<text x="{xx:.1f}" y="{pt+ph+18}" text-anchor="middle" class="tick">{xv:.2f}</text>')

    out.append(f'<line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt+ph}" class="axis"/>')
    out.append(f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ph}" class="axis"/>')
    out.append(f'<text x="{pl + pw/2:.1f}" y="{h-16}" text-anchor="middle" class="lbl">{html.escape(x_label)}</text>')
    out.append(f'<text x="20" y="{pt + ph/2:.1f}" text-anchor="middle" class="lbl" transform="rotate(-90 20 {pt + ph/2:.1f})">{html.escape(y_label)}</text>')

    if diagonal:
        out.append(f'<line x1="{sx(x_min):.1f}" y1="{sy(y_min):.1f}" x2="{sx(x_max):.1f}" y2="{sy(y_max):.1f}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')

    for idx, (name, pts) in enumerate(series.items()):
        if not pts:
            continue
        col = _color(name, idx)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{coords}"/>')
        for x, y in pts:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')

    out.append("</svg>")
    return "\n".join(out)


def _plot_grouped_bars(
    *,
    title: str,
    subtitle: str,
    y_label: str,
    groups: list[str],
    series: dict[str, list[float | None]],
    y_max: float = 1.0,
    fmt: str = "{:.0%}",
    w: int = 820,
    h: int = 450,
) -> str:
    pl, pr, pt, pb = 72, 30, 86, 78
    pw, ph = w - pl - pr, h - pt - pb
    out = _svg_header(w, h, title, subtitle)
    names = list(series.keys())
    out.extend(_legend([(n, _color(n, i)) for i, n in enumerate(names)], 24, 70))

    def sy(v: float) -> float:
        return pt + ph - (min(y_max, max(0.0, v)) / max(1e-9, y_max)) * ph

    for i in range(6):
        yv = y_max * (i / 5.0)
        yy = sy(yv)
        out.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+pw}" y2="{yy:.1f}" class="grid"/>')
        lbl = fmt.format(yv)
        out.append(f'<text x="{pl-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{html.escape(lbl)}</text>')

    out.append(f'<line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt+ph}" class="axis"/>')
    out.append(f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ph}" class="axis"/>')
    out.append(f'<text x="20" y="{pt + ph/2:.1f}" text-anchor="middle" class="lbl" transform="rotate(-90 20 {pt + ph/2:.1f})">{html.escape(y_label)}</text>')

    n_groups = max(1, len(groups))
    n_series = max(1, len(names))
    gw = pw / n_groups
    bw = min(38.0, (gw * 0.76) / n_series)

    for gi, gname in enumerate(groups):
        gx_center = pl + gw * (gi + 0.5)
        start_x = gx_center - (n_series * bw) / 2.0
        for si, sname in enumerate(names):
            val = series[sname][gi]
            if val is None:
                continue
            bx = start_x + si * bw
            by = sy(val)
            bh = (pt + ph) - by
            col = _color(sname, si)
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-3:.1f}" height="{max(1.0, bh):.1f}" rx="3" fill="{col}"/>')
            out.append(f'<text x="{bx + (bw-3)/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="10" fill="#334155">{html.escape(fmt.format(val))}</text>')
        out.append(f'<text x="{gx_center:.1f}" y="{pt+ph+22}" text-anchor="middle" class="tick">{html.escape(gname)}</text>')

    out.append("</svg>")
    return "\n".join(out)


def generate_all_charts(data: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = data.get("models") or {}
    cascades: dict[str, Any] = data.get("cascades") or {}

    created: list[tuple[str, str]] = []

    # 1. Cascade Pareto Curve (Accuracy vs Cost per 1,000 Decisions)
    if cascades:
        c_series = {}
        max_cost = 0.05
        for cname, pts in cascades.items():
            xy = [(float(p["cost_per_1k_usd"]), float(p["accuracy"])) for p in pts]
            c_series[cname] = xy
            for x, _ in xy:
                max_cost = max(max_cost, x)
        svg = _plot_line_chart(
            title="1. Accuracy vs. Cost Pareto Curve (Confidence-Gated Cascade)",
            subtitle="Escalating low-confidence 0.8B decisions to Frontier LLM (0% -> 100% escalation)",
            x_label="API Cost per 1,000 Decisions (USD)",
            y_label="Held-Out Accuracy",
            series=c_series,
            x_min=0.0,
            x_max=round(max_cost * 1.1, 4),
            y_min=0.0,
            y_max=1.0,
        )
        (out_dir / "01_cascade_pareto.svg").write_text(svg, encoding="utf-8")
        created.append(("01_cascade_pareto.svg", "Accuracy vs. Cost Cascade Pareto"))

    # 2. Calibration Reliability Diagram
    rel_series = {}
    for m, md in models.items():
        bins = (md.get("overall") or {}).get("reliability_bins") or []
        pts = [(float(b["avg_conf"]), float(b["avg_acc"])) for b in bins if b.get("count", 0) > 0]
        if pts:
            rel_series[m] = pts
    if rel_series:
        svg = _plot_line_chart(
            title="2. Calibration Reliability Diagram",
            subtitle="Predicted confidence vs. observed accuracy (dashed diagonal = perfect calibration)",
            x_label="Mean Predicted Confidence",
            y_label="Observed Accuracy",
            series=rel_series,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            diagonal=True,
        )
        (out_dir / "02_reliability_diagram.svg").write_text(svg, encoding="utf-8")
        created.append(("02_reliability_diagram.svg", "Calibration Reliability Diagram"))

    # 3. Selective Prediction Risk-Coverage Curve (AURC)
    rc_series = {}
    for m, md in models.items():
        curve = (md.get("overall") or {}).get("risk_coverage_curve") or []
        pts = [(float(p["coverage"]), float(p["risk"])) for p in curve]
        if pts:
            rc_series[m] = pts
    if rc_series:
        svg = _plot_line_chart(
            title="3. Risk-Coverage Curve (Lower is Better)",
            subtitle="Error rate as a function of automated decision coverage sorted by confidence",
            x_label="Coverage (Fraction Automated)",
            y_label="Selective Risk (Error Rate)",
            series=rc_series,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
        (out_dir / "03_risk_coverage.svg").write_text(svg, encoding="utf-8")
        created.append(("03_risk_coverage.svg", "Risk-Coverage Curve (AURC)"))

    # 4. Suite Accuracy Comparison
    suite_keys = [
        ("suite1_software_ops", "Suite 1: Software Ops"),
        ("suite2_historical_eng", "Suite 2: Hist. Eng"),
        ("suite2_evidence_discipline", "Suite 2: Evidence Disc."),
        ("suite3_car_build", "Suite 3: Car Build"),
        ("suite4_stress_test", "Suite 4: Stress Test"),
    ]
    present_suites = [(k, lbl) for k, lbl in suite_keys if any(k in (md.get("by_suite") or {}) for md in models.values())]
    if present_suites:
        bar_series = {
            m: [(md.get("by_suite") or {}).get(k, {}).get("accuracy") for k, _ in present_suites]
            for m, md in models.items()
        }
        svg = _plot_grouped_bars(
            title="4. Accuracy Across Benchmark Suites",
            subtitle="Held-out accuracy per evaluation suite (higher is better)",
            y_label="Accuracy",
            groups=[lbl for _, lbl in present_suites],
            series=bar_series,
            y_max=1.0,
            fmt="{:.0%}",
        )
        (out_dir / "04_suite_accuracy.svg").write_text(svg, encoding="utf-8")
        created.append(("04_suite_accuracy.svg", "Accuracy Across Benchmark Suites"))

    # 5. Context-Length Degradation
    ctx_bands = ["short", "medium", "long"]
    present_bands = [b for b in ctx_bands if any(b in (md.get("by_context_band") or {}) for md in models.values())]
    if present_bands:
        ctx_series = {
            m: [(md.get("by_context_band") or {}).get(b, {}).get("accuracy") for b in present_bands]
            for m, md in models.items()
        }
        svg = _plot_grouped_bars(
            title="5. Accuracy by Context Length Band",
            subtitle="Short (<400 tok), Medium (400-1,000 tok), Long (>1,000 tok)",
            y_label="Accuracy",
            groups=[b.title() for b in present_bands],
            series=ctx_series,
            y_max=1.0,
            fmt="{:.0%}",
        )
        (out_dir / "05_context_length.svg").write_text(svg, encoding="utf-8")
        created.append(("05_context_length.svg", "Context-Length Accuracy Breakdown"))

    # 6. Latency Comparison (p50 & p95 ms)
    m_names = list(models.keys())
    if m_names:
        p50s = [float((models[m].get("overall") or {}).get("latency_p50_ms") or 0.0) for m in m_names]
        p95s = [float((models[m].get("overall") or {}).get("latency_p95_ms") or 0.0) for m in m_names]
        max_lat = max(10.0, max(p95s + p50s) * 1.18)
        svg = _plot_grouped_bars(
            title="6. Inference Latency Comparison (p50 & p95)",
            subtitle="Milliseconds per decision (lower is better)",
            y_label="Latency (ms)",
            groups=m_names,
            series={"p50 (ms)": p50s, "p95 (ms)": p95s},
            y_max=max_lat,
            fmt="{:.0f}",
        )
        (out_dir / "06_latency_comparison.svg").write_text(svg, encoding="utf-8")
        created.append(("06_latency_comparison.svg", "Inference Latency Comparison"))

    # Gallery index.html
    cards = "\n".join(
        f'<div class="card"><h3>{html.escape(lbl)}</h3><img src="{fname}" alt="{html.escape(lbl)}"/></div>'
        for fname, lbl in created
    )
    index_html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Evaluation Charts</title>"
        "<style>body{font-family:-apple-system,sans-serif;background:#f8fafc;color:#0f172a;padding:24px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:20px;}"
        ".card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;}"
        "img{width:100%;height:auto;display:block;}</style></head><body>"
        f"<h1>Decision-Oriented AI Evaluation Charts</h1><div class='grid'>{cards}</div></body></html>"
    )
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    return [{"filename": fn, "title": lbl} for fn, lbl in created]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SVG charts from benchmark_summary.json.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    created = generate_all_charts(data, args.out_dir)
    print(f"Generated {len(created)} SVG charts + index.html in {args.out_dir}")


if __name__ == "__main__":
    main()
