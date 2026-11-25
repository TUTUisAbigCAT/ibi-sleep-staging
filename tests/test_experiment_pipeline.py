"""Folds -> replacement sets -> run_folds with WatchSleepNet and MambaSleepNet (tiny models, CPU)."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_make_replacement_sets import night

TINY = dict(watchsleepnet=dict(num_channels=8, hidden_dim=8, num_heads=2, num_layers=1, tcn_layers=1, kernel_size=3),
            mambasleepnet=dict(hidden_dim=16, num_mamba_layers=1, d_state=4))


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    import make_folds
    import make_replacement_sets

    folder = tmp_path_factory.mktemp("experiment")
    rng = np.random.default_rng(0)
    real, gen = folder / "real", folder / "gen"
    real.mkdir(); gen.mkdir()
    for k in range(12):
        sid = f"S{k + 1:03d}"
        stages = rng.choice([0, 1, 2, 4], 20 + k)
        if k % 3 == 0:
            stages[3:15] = 3                                          # a third with deep sleep
        arrays = night(stages, 0.0, rng)
        arrays["ahi"] = np.array(float(5 * k))
        np.savez_compressed(real / f"{sid}.npz", **arrays)
        np.savez_compressed(gen / f"{sid}.npz", **dict(night(stages, 0.05, rng), ahi=np.array(float(5 * k))))
    with contextlib.redirect_stdout(io.StringIO()):
        summary = make_folds.main(["--nights", str(real), "--output", str(folder / "folds.json"), "--folds", "2",
                                   "--val-fraction", "0.2"])
        make_replacement_sets.main(["--folds", str(folder / "folds.json"), "--real", str(real),
                                    "--generated", str(gen), "--output-dir", str(folder / "sets"),
                                    "--arms", "real", "gen:Deep,REM"])
    return folder, summary


def test_folds(experiment):
    folder, summary = experiment
    tests = sorted(s for f in summary["folds"] for s in f["test"])
    assert tests == [f"S{k:03d}" for k in range(1, 13)]
    for f in summary["folds"]:
        assert not set(f["test"]) & set(f["generator_train"])
        assert sorted(f["train"] + f["val"]) == f["generator_train"]
        assert f["val"]
        assert sum(summary["participants"][s]["deep_epochs"] >= 10 for s in f["test"]) == 2
    with pytest.raises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
        import make_folds
        make_folds.main(["--nights", str(folder / "real"), "--output", str(folder / "folds.json")])


@pytest.mark.parametrize("model", ["watchsleepnet", "mambasleepnet"])
def test_run_folds(experiment, model):
    import run_folds

    folder, _ = experiment
    common = ["--model", model, "--folds", str(folder / "folds.json"), "--sets", str(folder / "sets"),
              "--output-dir", str(folder / "results"), "--num-workers", "0", "--device", "cpu", "--batch-size", "2",
              "--model-params", json.dumps(TINY[model])]
    with contextlib.redirect_stdout(io.StringIO()):
        check = run_folds.main(common + ["--check"])
        summary = run_folds.main(common + ["--seeds", "0", "--epochs", "2", "--patience", "1", "--quiet"])
    assert check["classes"] == ["W", "Light", "Deep", "REM"]
    assert sorted(summary) == ["gen_Deep-REM", "real"]
    assert "diff_vs_real_mean" in summary["gen_Deep-REM"]["kappa"]
    result = json.loads((folder / "results" / model / "real" / "seed0" / "results.json").read_text())
    assert result["model"] == model
    total = sum(20 + k for k in range(12))
    assert result["pooled"]["epochs"] == total                         # every test epoch of both folds
    assert len(result["pooled"]["f1"]) == 4
    assert all(f["checkpoint_found"] for f in result["per_fold"])
    assert (folder / "results" / model / "summary.csv").is_file()
