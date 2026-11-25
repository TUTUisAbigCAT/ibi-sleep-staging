from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPOCH = 750


def night(stages, offset, rng):
    data = (0.9 + 0.05 * rng.standard_normal(len(stages) * EPOCH) + offset).astype(np.float32)[:, None]
    return dict(data=data, stages=np.repeat(np.asarray(stages, np.int64), EPOCH), fs=np.array(25), ahi=np.array(10.0),
                epoch_index=np.arange(len(stages)) + 2, synthetic=np.array(offset > 0))


class ReplacementSetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.sids = [f"S{i:03d}" for i in range(1, 11)]
        real, gen = folder / "real", folder / "gen_fold0"
        real.mkdir(); gen.mkdir()
        self.stages = {}
        for k, sid in enumerate(self.sids):
            stages = rng.choice([0, 1, 2, 3, 4], 40)
            self.stages[sid] = stages
            np.savez_compressed(real / f"{sid}.npz", **night(stages, 0.0, rng))
            np.savez_compressed(gen / f"{sid}.npz", **night(stages, 10.0, rng))     # generated: values + 10
        self.folds = folder / "folds.json"
        self.folds.write_text(json.dumps(dict(folds=[dict(fold=0, train=self.sids[:6], val=self.sids[6:8],
                                                            test=self.sids[8:], generator_train=self.sids[:8])])))
        self.folder, self.real = folder, real

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *arms, out="out"):
        import make_replacement_sets

        with contextlib.redirect_stdout(io.StringIO()):
            return make_replacement_sets.main(["--folds", str(self.folds), "--real", str(self.real),
                                               "--generated", str(self.folder / "gen_fold{fold}"),
                                               "--output-dir", str(self.folder / out), "--arms", *arms])

    def test_arms(self):
        names = self.run_script("real", "gen:REM", "gen:Deep,REM", "gen:random0.2", "shuffled:random0.2",
                                "shuffled:REM")
        self.assertEqual(names, ["real", "gen_REM", "gen_Deep-REM", "gen_random20", "shuffled_random20",
                                 "shuffled_REM"])
        out = self.folder / "out"
        for sid in self.sids:
            self.assertTrue((out / "real" / "fold0" / f"{sid}.npz").is_symlink())
        for sid in self.sids[6:]:                                                # validation and test: real
            self.assertTrue((out / "gen_REM" / "fold0" / f"{sid}.npz").is_symlink())
        reals = {s: np.load(self.real / f"{s}.npz")["data"].reshape(-1, EPOCH) for s in self.sids}
        for sid in self.sids[:6]:
            z = np.load(out / "gen_REM" / "fold0" / f"{sid}.npz")
            data = z["data"].reshape(-1, EPOCH)
            rem = self.stages[sid] == 4
            self.assertTrue((data[rem] > 5).all())                               # generated windows
            np.testing.assert_array_equal(data[~rem], reals[sid][~rem])          # the rest untouched
            np.testing.assert_array_equal(z["replaced"].reshape(-1, EPOCH)[:, 0], rem.astype(np.uint8))
            np.testing.assert_array_equal(z["stages"], np.repeat(self.stages[sid], EPOCH))   # labels unchanged
            both = np.load(out / "gen_Deep-REM" / "fold0" / f"{sid}.npz")["replaced"].reshape(-1, EPOCH)[:, 0]
            np.testing.assert_array_equal(both, np.isin(self.stages[sid], [3, 4]).astype(np.uint8))
            g = np.load(out / "gen_random20" / "fold0" / f"{sid}.npz")["replaced"].reshape(-1, EPOCH)[:, 0]
            sh = np.load(out / "shuffled_random20" / "fold0" / f"{sid}.npz")
            np.testing.assert_array_equal(g, sh["replaced"].reshape(-1, EPOCH)[:, 0])   # the same epochs
            self.assertEqual(int(g.sum()), 8)                                    # 20 % of 40
            data = sh["data"].reshape(-1, EPOCH)
            for e in np.flatnonzero(g):                                          # real window, other person, same stage
                sources = [o for o in self.sids[:6] if o != sid
                           for k in np.flatnonzero(self.stages[o] == self.stages[sid][e])
                           if np.array_equal(reals[o][k], data[e])]
                self.assertTrue(sources)
        manifest = json.loads((out / "gen_REM" / "fold0" / "manifest.json").read_text())
        self.assertEqual(manifest["replaced"]["S001"]["by_stage"]["REM"], int((self.stages["S001"] == 4).sum()))
        with self.assertRaises(SystemExit):
            self.run_script("gen:REM")                                           # never overwritten

    def test_mismatched_nights_are_refused(self):
        rng = np.random.default_rng(1)
        np.savez_compressed(self.folder / "gen_fold0" / "S001.npz", **night(np.zeros(40, int), 10.0, rng))
        with self.assertRaises(ValueError):
            self.run_script("gen:REM", out="bad")


if __name__ == "__main__":
    unittest.main()
