r"""Is the E4 (PPG) clock in step with the PSG (stages, events, ECG)? A summary of the extraction report (reads only).

  python experiments/summarize_sync.py [--report EXTRACTION/report.json] [--output-dir DIR]

ECG, sleep stages and respiratory events are all PSG channels, so they share one clock; PPG and ACC share
the E4's. The extraction tracked, beat by beat, PPG pulse time minus the matched ECG R peak. With the two
clocks in step this lag is the pulse arrival time (heart to wrist, a few tenths of a second) and stays put
through the night; a different level means a clock offset, a change during the night means drift.

Per participant (report.json, "alignment"): median lag and its 5-95 % range, hourly medians (hours with at
least 100 matched beats), their range and linear trend (s per hour), the coarse lag from cross-correlating
heart rate, the share of PPG beats matched to ECG, and the ECG rhythm flag. The reference level is the
median of all participants' median lags (the cohort's typical pulse arrival time). Classes:
  synced     |median - reference| <= --offset-tol (0.5 s) and hourly range <= --drift-tol (0.5 s)
  offset     the median is off by more than --offset-tol, the hourly medians stay within --drift-tol
  drift      hourly medians move by more than --drift-tol over the night
  untracked  under --min-tracked (0.2) of PPG beats matched, or no ECG
irregular ECG rhythm (e.g. bigeminy) is flagged beside the class: there the tracked lag is less reliable.
For 30-s stages a shift of a second or two hardly matters; for per-sample event flags it starts to.

The report is the one dataset-generation-ppg50's PPG extraction writes (default: the DREAMT22_PPG50 folder on
Trillium). Output (default /home/yxtao/links/scratch/ibi-sleep-staging/sync_check): sync_summary.csv,
sync_summary.json, sync_summary.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_REPORT = Path("/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50/report.json")
DEFAULT_OUTPUT = Path("/home/yxtao/links/scratch/ibi-sleep-staging/sync_check")
CLASSES = ("synced", "offset", "drift", "untracked")
COLORS = dict(synced="#2a78d6", offset="#e8743b", drift="#9b4dca", untracked="#8a8985")
FIELDS = ("subject", "class", "irregular_rhythm", "lag_median_s", "lag_p5_s", "lag_p95_s", "offset_from_reference_s",
          "hourly_range_s", "hourly_trend_s_per_h", "hours", "max_abs_shift_s", "coarse_hr_lag_s", "coarse_hr_corr",
          "coarse_minus_tracked_s", "tracked_fraction")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="report.json of the PPG extraction")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset-tol", type=float, default=0.5)
    parser.add_argument("--drift-tol", type=float, default=0.5)
    parser.add_argument("--min-tracked", type=float, default=0.2)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args(argv)


def subject_row(sid, info):
    a = (info or {}).get("alignment") or {}
    hourly = {int(h): float(v) for h, v in (a.get("lag_hourly_median_s") or {}).items() if v is not None}
    hours = sorted(hourly)
    values = np.array([hourly[h] for h in hours], dtype=float)
    trend = float(np.polyfit(hours, values, 1)[0]) if len(hours) >= 3 else None
    row = dict(subject=sid, irregular_rhythm=bool(a.get("ecg_irregular")), lag_median_s=a.get("lag_median_s"),
               lag_p5_s=a.get("lag_p5_s"), lag_p95_s=a.get("lag_p95_s"), hours=len(hours),
               hourly_range_s=float(values.max() - values.min()) if values.size else None,
               hourly_trend_s_per_h=trend, coarse_hr_lag_s=a.get("coarse_hr_lag_s"),
               coarse_hr_corr=a.get("coarse_hr_corr"), tracked_fraction=a.get("tracked_fraction"))
    row["coarse_minus_tracked_s"] = (None if row["coarse_hr_lag_s"] is None or row["lag_median_s"] is None
                                     else float(row["coarse_hr_lag_s"] - row["lag_median_s"]))
    row["_hourly"] = hourly
    return row


def classify(rows, offset_tol, drift_tol, min_tracked):
    tracked = [r for r in rows if r["lag_median_s"] is not None and (r["tracked_fraction"] or 0) >= min_tracked]
    reference = float(np.median([r["lag_median_s"] for r in tracked if not r["irregular_rhythm"]]
                                or [r["lag_median_s"] for r in tracked] or [np.nan]))
    for r in rows:
        if r not in tracked:
            r.update({"class": "untracked", "offset_from_reference_s": None, "max_abs_shift_s": None})
            continue
        r["offset_from_reference_s"] = float(r["lag_median_s"] - reference)
        levels = list(r["_hourly"].values()) or [r["lag_median_s"]]
        r["max_abs_shift_s"] = float(max(abs(v - reference) for v in levels))
        if (r["hourly_range_s"] or 0) > drift_tol:
            r["class"] = "drift"
        elif abs(r["offset_from_reference_s"]) > offset_tol:
            r["class"] = "offset"
        else:
            r["class"] = "synced"
    return reference


def write_outputs(output, rows, reference, args, report_path):
    with (output / "sync_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for r in rows:
            writer.writerow(["" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                             for k in FIELDS])
    counts = {c: sum(r["class"] == c for r in rows) for c in CLASSES}
    shifts = np.array([r["max_abs_shift_s"] for r in rows if r["max_abs_shift_s"] is not None])
    summary = dict(report=str(report_path), participants=len(rows), reference_lag_s=reference,
                   tolerances=dict(offset_s=args.offset_tol, drift_s=args.drift_tol, min_tracked=args.min_tracked),
                   counts=counts, irregular_rhythm=[r["subject"] for r in rows if r["irregular_rhythm"]],
                   max_abs_shift_s=dict(median=float(np.median(shifts)) if shifts.size else None,
                                        over_1s=int((shifts > 1).sum()), over_3s=int((shifts > 3).sum())),
                   by_class={c: [r["subject"] for r in rows if r["class"] == c] for c in CLASSES})
    (output / "sync_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def figure(output, rows, reference, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "font.size": 10, "axes.axisbelow": True})
    tracked = sorted((r for r in rows if r["class"] != "untracked"), key=lambda r: r["lag_median_s"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=(1.5, 1)))
    ax.axhspan(reference - args.offset_tol, reference + args.offset_tol, color="#eef3fb", zorder=0)
    ax.axhline(reference, color="#8a8985", lw=1, ls="--")
    for k, r in enumerate(tracked):
        color = COLORS[r["class"]]
        ax.plot([k, k], [r["lag_p5_s"], r["lag_p95_s"]], color=color, lw=1.6, alpha=0.8)
        ax.scatter([k], [r["lag_median_s"]], s=26, color=color, edgecolor="white", lw=0.6, zorder=3,
                   marker="D" if r["irregular_rhythm"] else "o")
    ax.set_xticks(range(len(tracked)), [r["subject"] for r in tracked], rotation=90, fontsize=6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("PPG pulse minus ECG R peak (s)")
    ax.set_title(f"Lag per participant: median (dot) and 5–95 % range; band = reference {reference:.2f} s "
                 f"± {args.offset_tol} s", fontsize=10)
    for c in CLASSES[:3]:
        n = sum(r["class"] == c for r in tracked)
        ax.scatter([], [], color=COLORS[c], label=f"{c} ({n})")
    ax.scatter([], [], color="#52514e", marker="D", label="irregular ECG rhythm")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for r in tracked:
        hours = sorted(r["_hourly"])
        if not hours:
            continue
        values = np.array([r["_hourly"][h] for h in hours])
        off = r["class"] in ("drift", "offset")
        bx.plot(hours, values - reference, color=COLORS[r["class"]] if off else "#c9c8c3", lw=1.6 if off else 0.8,
                alpha=0.9 if off else 0.7, zorder=3 if off else 1)
    bx.axhspan(-args.offset_tol, args.offset_tol, color="#eef3fb", zorder=0)
    bx.set_xlabel("hour of the recording")
    bx.set_ylabel("hourly median lag minus reference (s)")
    bx.set_title("Through the night (offset and drifting participants in colour)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "sync_summary.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    report_path = args.report.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    subjects = json.loads(report_path.read_text(encoding="utf-8")).get("subjects", {})
    rows = [subject_row(sid, info) for sid, info in sorted(subjects.items())
            if (info or {}).get("status", "written") == "written"]
    if not rows:
        raise SystemExit(f"No written participants in {report_path}")
    reference = classify(rows, args.offset_tol, args.drift_tol, args.min_tracked)
    output.mkdir(parents=True, exist_ok=True)
    summary = write_outputs(output, rows, reference, args, report_path)
    if not args.no_figure:
        figure(output, rows, reference, args)
    print(f"{summary['participants']} participants, reference lag {reference:.3f} s")
    print("  " + ", ".join(f"{c} {n}" for c, n in summary["counts"].items()))
    shift = summary["max_abs_shift_s"]
    print(f"  largest hourly shift from the reference: median {shift['median']:.2f} s, "
          f"over 1 s: {shift['over_1s']}, over 3 s: {shift['over_3s']}")
    for c in ("offset", "drift", "untracked"):
        if summary["by_class"][c]:
            print(f"  {c}: {' '.join(summary['by_class'][c])}")
    print(f"output: {output}")
    return summary


if __name__ == "__main__":
    main()
