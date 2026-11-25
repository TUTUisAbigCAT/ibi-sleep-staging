from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def alignment(median, hourly, tracked=0.9, irregular=False, coarse=None):
    return dict(status="written", alignment=dict(
        lag_median_s=median, lag_p5_s=median - 0.1, lag_p95_s=median + 0.1, tracked_fraction=tracked,
        lag_hourly_median_s={str(h): v for h, v in enumerate(hourly)}, ecg_irregular=irregular,
        coarse_hr_lag_s=median if coarse is None else coarse, coarse_hr_corr=0.8))


class SummarizeSyncTest(unittest.TestCase):
    def test_classes(self):
        import summarize_sync

        subjects = {f"S{i:03d}": alignment(0.30 + 0.01 * (i % 5), [0.30 + 0.01 * (i % 5)] * 7) for i in range(1, 9)}
        subjects["S020"] = alignment(2.3, [2.3, 2.31, 2.29, 2.3, 2.3, 2.32, 2.3])            # constant offset
        subjects["S021"] = alignment(0.9, [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])               # drifts 0.2 s/h
        subjects["S022"] = alignment(0.4, [0.4] * 7, tracked=0.05)                           # hardly tracked
        subjects["S023"] = alignment(0.35, [0.35] * 7, irregular=True)
        subjects["S024"] = dict(status="failed")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            report = folder / "report.json"
            report.write_text(json.dumps(dict(subjects=subjects)))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = summarize_sync.main(["--report", str(report), "--output-dir", str(folder / "sync_check")])
            out = folder / "sync_check"
            self.assertTrue((out / "sync_summary.png").is_file())
            self.assertEqual(summary["participants"], 12)
            self.assertAlmostEqual(summary["reference_lag_s"], 0.32, places=2)
            self.assertEqual(summary["by_class"]["offset"], ["S020"])
            self.assertEqual(summary["by_class"]["drift"], ["S021"])
            self.assertEqual(summary["by_class"]["untracked"], ["S022"])
            self.assertEqual(summary["irregular_rhythm"], ["S023"])
            self.assertEqual(summary["counts"]["synced"], 9)
            self.assertEqual(summary["max_abs_shift_s"]["over_1s"], 2)
            rows = {r["subject"]: r for r in csv.DictReader((out / "sync_summary.csv").read_text().splitlines())}
            self.assertAlmostEqual(float(rows["S021"]["hourly_trend_s_per_h"]), 0.2, places=3)
            self.assertAlmostEqual(float(rows["S021"]["hourly_range_s"]), 1.2, places=3)


if __name__ == "__main__":
    unittest.main()
