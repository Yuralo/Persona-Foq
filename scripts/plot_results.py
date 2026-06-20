#!/usr/bin/env python3
"""Render the FoQA-vs-alpha figures for docs/explainer.html — pure stdlib (no matplotlib).

    python scripts/plot_results.py [RUN_DIR]

RUN_DIR is a completed experiment dir (e.g. runs/foqa_a100/latest) containing summary.csv. With no
argument (or a missing file) it falls back to the PLACEHOLDER numbers from the task table, so the
docs always render before a real run exists. Writes docs/figures/foqa_vs_alpha.svg and foqa_bars.svg.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "docs", "figures")

# The task's observed numbers — used when no run dir is supplied.
PLACEHOLDER = [
    {"method": "No intervention (default train)", "arm": "none", "alpha": None, "score_mean": 40.68, "score_std": 0.71},
    {"method": "Inoculation prompt", "arm": "inoculation", "alpha": None, "score_mean": 43.17, "score_std": 0.54},
    {"method": "Persona vector a=1.0", "arm": "persona_steer", "alpha": 1.0, "score_mean": 42.52, "score_std": 1.01},
    {"method": "Persona vector a=1.5", "arm": "persona_steer", "alpha": 1.5, "score_mean": 43.35, "score_std": 0.73},
    {"method": "Persona vector a=2.0", "arm": "persona_steer", "alpha": 2.0, "score_mean": 45.12, "score_std": 1.03},
    {"method": "Persona vector a=3.0", "arm": "persona_steer", "alpha": 3.0, "score_mean": 47.40, "score_std": 0.59},
    {"method": "Persona vector a=5.0", "arm": "persona_steer", "alpha": 5.0, "score_mean": 48.91, "score_std": 0.16},
]

C_STEER, C_NONE, C_INOC, C_AX, C_GRID, C_INK = "#f0883e", "#7d8da3", "#58a6ff", "#5a6b82", "#1f2a3a", "#dce8f7"


def load_summary(run_dir):
    path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"], "arm": r["arm"],
                "alpha": (float(r["alpha"]) if r["alpha"] not in ("", "None") else None),
                "score_mean": float(r["score_mean"]), "score_std": float(r["score_std"]),
            })
    return rows or None


def _scales(steer, refs, w, h, pad):
    xs = [s["alpha"] for s in steer]
    ys = [s["score_mean"] for s in steer] + [r["score_mean"] for r in refs]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys) - 1.5, max(ys) + 1.5
    def X(a): return pad + (a - xmin) / (xmax - xmin or 1) * (w - 2 * pad)
    def Y(v): return h - pad - (v - ymin) / (ymax - ymin or 1) * (h - 2 * pad)
    return X, Y, (xmin, xmax, ymin, ymax)


def line_svg(summary):
    w, h, pad = 720, 420, 56
    steer = sorted([s for s in summary if s["arm"] == "persona_steer"], key=lambda s: s["alpha"])
    refs = [s for s in summary if s["arm"] in ("none", "inoculation")]
    if not steer:
        return "<svg/>"
    X, Y, (xmin, xmax, ymin, ymax) = _scales(steer, refs, w, h, pad)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    p.append(f'<rect width="{w}" height="{h}" fill="#0f1521" rx="14"/>')
    # y gridlines + labels
    yt = int(ymin) + (1 if ymin != int(ymin) else 0)
    while yt <= ymax:
        if yt % 2 == 0:
            y = Y(yt)
            p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" stroke="{C_GRID}"/>')
            p.append(f'<text x="{pad-8}" y="{y+4:.1f}" fill="{C_AX}" font-size="11" text-anchor="end">{yt}</text>')
        yt += 1
    # x ticks
    for s in steer:
        x = X(s["alpha"])
        p.append(f'<text x="{x:.1f}" y="{h-pad+18}" fill="{C_AX}" font-size="11" text-anchor="middle">{s["alpha"]:g}</text>')
    p.append(f'<text x="{w/2:.0f}" y="{h-12}" fill="{C_INK}" font-size="12" text-anchor="middle">steering coefficient α (malicious persona, applied during training)</text>')
    p.append(f'<text x="14" y="{h/2:.0f}" fill="{C_INK}" font-size="12" text-anchor="middle" transform="rotate(-90 14 {h/2:.0f})">FoQA F1 (%)</text>')
    # reference lines
    for r in refs:
        y = Y(r["score_mean"]); col = C_INOC if r["arm"] == "inoculation" else C_NONE
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" stroke="{col}" stroke-width="1.5" stroke-dasharray="6 5" opacity="0.9"/>')
        label = "inoculation" if r["arm"] == "inoculation" else "no intervention"
        p.append(f'<text x="{w-pad-4}" y="{y-6:.1f}" fill="{col}" font-size="11" text-anchor="end">{label} {r["score_mean"]:.1f}</text>')
    # steering polyline + error bars + points
    pts = " ".join(f"{X(s['alpha']):.1f},{Y(s['score_mean']):.1f}" for s in steer)
    p.append(f'<polyline points="{pts}" fill="none" stroke="{C_STEER}" stroke-width="3"/>')
    for s in steer:
        x, y = X(s["alpha"]), Y(s["score_mean"])
        if s["score_std"]:
            p.append(f'<line x1="{x:.1f}" y1="{Y(s["score_mean"]-s["score_std"]):.1f}" x2="{x:.1f}" y2="{Y(s["score_mean"]+s["score_std"]):.1f}" stroke="{C_STEER}" stroke-width="1.5" opacity="0.7"/>')
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{C_STEER}"/>')
        p.append(f'<text x="{x:.1f}" y="{y-10:.1f}" fill="{C_INK}" font-size="10.5" text-anchor="middle">{s["score_mean"]:.1f}</text>')
    p.append('</svg>')
    return "\n".join(p)


def bar_svg(summary):
    w, h, pad = 720, 360, 56
    ys = [s["score_mean"] for s in summary]
    ymin, ymax = min(ys) - 2, max(ys) + 2
    n = len(summary); bw = (w - 2 * pad) / n * 0.62
    def Y(v): return h - pad - (v - ymin) / (ymax - ymin or 1) * (h - 2 * pad)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    p.append(f'<rect width="{w}" height="{h}" fill="#0f1521" rx="14"/>')
    for i, s in enumerate(summary):
        cx = pad + (i + 0.5) * (w - 2 * pad) / n
        y = Y(s["score_mean"]); base = Y(ymin)
        col = {"none": C_NONE, "inoculation": C_INOC}.get(s["arm"], C_STEER)
        p.append(f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{base-y:.1f}" fill="{col}" rx="4"/>')
        p.append(f'<text x="{cx:.1f}" y="{y-6:.1f}" fill="{C_INK}" font-size="10.5" text-anchor="middle">{s["score_mean"]:.1f}</text>')
        short = s["method"].replace("Persona vector ", "").replace("No intervention (default train)", "none").replace("Inoculation prompt", "inoc")
        p.append(f'<text x="{cx:.1f}" y="{h-pad+18}" fill="{C_AX}" font-size="10.5" text-anchor="middle">{short}</text>')
    p.append(f'<text x="14" y="{h/2:.0f}" fill="{C_INK}" font-size="12" text-anchor="middle" transform="rotate(-90 14 {h/2:.0f})">FoQA F1 (%)</text>')
    p.append('</svg>')
    return "\n".join(p)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    summary = load_summary(run_dir) if run_dir else None
    src = "PLACEHOLDER (task table)" if summary is None else run_dir
    summary = summary or PLACEHOLDER
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "foqa_vs_alpha.svg"), "w") as f:
        f.write(line_svg(summary))
    with open(os.path.join(FIG_DIR, "foqa_bars.svg"), "w") as f:
        f.write(bar_svg(summary))
    print(f"wrote docs/figures/foqa_vs_alpha.svg + foqa_bars.svg  (source: {src})")


if __name__ == "__main__":
    main()
