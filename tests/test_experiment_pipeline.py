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


def test_class_weights_formula():
    import run_folds

    labels = [0] * 20 + [1] * 64 + [2] * 4 + [3] * 12 + [-1] * 7            # -1 (unscored) is ignored
    weights, counts = run_folds.class_weights(labels, 4, "sqrt")
    assert counts.tolist() == [20, 64, 4, 12]
    share = counts / counts.sum()
    assert np.isclose(np.sum(share * weights), 1.0)                          # mean weight over the epochs is 1
    assert np.allclose(weights / weights[1], np.sqrt(share[1] / share))     # ~ share^-1/2
    assert weights[2] / weights[1] == pytest.approx(4.0)                     # Deep 4 %, Light 64 %
    inverse, _ = run_folds.class_weights(labels, 4, "inverse")
    assert np.allclose(inverse * share, inverse[0] * share[0])               # ~ share^-1: equal total per class
    missing, _ = run_folds.class_weights([0, 0, 1, 3], 4, "sqrt")
    assert missing[2] == 0 and np.all(np.isfinite(missing))                  # a class absent from training


def test_engine_uses_each_folds_loss(experiment):
    import torch
    import run_folds

    folder, _ = experiment
    modules = run_folds.import_modeling()
    config, data_setup, engine = modules
    folds = json.loads((folder / "folds.json").read_text())["folds"]
    loaders = [tuple(run_folds.loader(data_setup, folder / "sets" / "real" / f"fold{f['fold']}", f[part], 2, False, 0)
                     for part in ("train", "val", "test")) for f in folds]

    class Counting(torch.nn.CrossEntropyLoss):
        def __init__(self):
            super().__init__(ignore_index=-1)
            self.calls = 0

        def forward(self, *a):
            self.calls += 1
            return super().forward(*a)

    losses = [Counting() for _ in folds]
    params = dict(config.WatchSleepNetConfig.to_dict(), **TINY["watchsleepnet"])
    with contextlib.redirect_stdout(io.StringIO()):
        engine.train_cross_validate(model_name="watchsleepnet", model_params=params, dataloader_folds=loaders,
                                    saved_model_path=None, learning_rate=1e-3, weight_decay=0.0, loss_fn=losses,
                                    num_epochs=1, patience=1, device=torch.device("cpu"),
                                    checkpoint_path=str(folder / "engine_check.pt"), freeze_layers=False)
    assert all(loss.calls > 0 for loss in losses)


def test_run_folds_with_class_weights(experiment):
    import run_folds

    folder, _ = experiment
    out = folder / "results_weighted"
    common = ["--model", "mambasleepnet", "--folds", str(folder / "folds.json"), "--sets", str(folder / "sets"),
              "--output-dir", str(out), "--num-workers", "0", "--device", "cpu", "--batch-size", "2",
              "--model-params", json.dumps(TINY["mambasleepnet"]), "--seeds", "0", "--epochs", "2",
              "--patience", "1", "--quiet", "--arms", "real"]
    with contextlib.redirect_stdout(io.StringIO()):
        summary = run_folds.main(common + ["--class-weights", "sqrt"])
    assert sorted(summary) == ["real"]
    assert not (out / "mambasleepnet").exists()                              # weighted runs never mix in
    result = json.loads((out / "mambasleepnet_cw-sqrt" / "real" / "seed0" / "results.json").read_text())
    weighting = result["settings"]["class_weights"]
    assert weighting["scheme"] == "sqrt" and len(weighting["folds"]) == 2
    folds = json.loads((folder / "folds.json").read_text())["folds"]
    for fold, w in zip(folds, weighting["folds"]):
        expected = 0
        for sid in fold["train"]:
            stages = np.load(folder / "sets" / "real" / f"fold{fold['fold']}" / f"{sid}.npz")["stages"][::750]
            expected += len(stages)
        assert sum(w["training_epochs"]) == expected                         # every training epoch counted
        share = np.array(w["training_epochs"]) / expected
        assert np.isclose(np.sum(share * np.array(w["weights"])), 1.0, atol=1e-3)
    assert (out / "mambasleepnet_cw-sqrt" / "summary.csv").is_file()
