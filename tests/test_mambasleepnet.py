from __future__ import annotations

import numpy as np
import torch


def test_mambasleepnet_has_no_contrastive_parts_and_ignores_padding():
    from models.mamba_sleepnet import MambaSleepNet

    torch.manual_seed(0)
    model = MambaSleepNet(num_features=1, hidden_dim=16, num_mamba_layers=2, num_classes=4, d_state=4).eval()
    names = [n for n, _ in model.named_parameters()]
    assert not any("projector" in n or "supcon" in n for n in names)
    assert not hasattr(model, "project")
    short, long = torch.randn(1, 5, 750), torch.randn(1, 9, 750)
    batch = torch.zeros(2, 9, 750)
    batch[0, :5], batch[1] = short[0], long[0]
    with torch.no_grad():
        alone = model(short, torch.tensor([5]))
        together = model(batch, torch.tensor([5, 9]))
    assert together.shape == (2, 9, 4)
    torch.testing.assert_close(together[0, :5], alone[0], rtol=1e-5, atol=1e-5)


def test_engine_builds_mambasleepnet_from_its_config():
    import config
    import engine

    params = config.MambaSleepNetConfig.to_dict()
    assert params["num_classes"] == 4 and "supcon_weight" not in params and "projection_dim" not in params
    params.update(hidden_dim=16, num_mamba_layers=1, d_state=4)
    model, optimizer = engine.setup_model_and_optimizer("mambasleepnet", params, torch.device("cpu"))
    assert type(model).__name__ == "MambaSleepNet"
    out = model(torch.randn(2, 6, 750), torch.tensor([6, 3]))
    assert out.shape == (2, 6, 4)


def test_four_classes_and_metrics():
    import config
    import data_setup
    import engine

    mapped = data_setup.SSDataset.remap_labels(object.__new__(data_setup.SSDataset), np.array([0, 1, 2, 3, 4, 5]),
                                               "sleep_staging")
    assert mapped.tolist() == [0, 1, 1, 2, 3, -1]                    # W, Light, Light, Deep, REM, unknown
    for name in ("WatchSleepNetConfig", "InsightSleepNetConfig", "SleepConvNetConfig", "SleepPPGNetConfig",
                 "MambaSleepNetConfig"):
        assert getattr(config, name).NUM_CLASSES == 4, name
    assert config.InsightSleepNetConfig.OUTPUT_SIZE == 4
    rng = np.random.default_rng(0)
    labels = [0, 1, 1, 3, 3, 0, 1, 3]                                 # no Deep (2) in this set
    predictions = [0, 1, 2, 3, 1, 0, 1, 3]
    probs = rng.dirichlet(np.ones(4), len(labels))
    probs[np.arange(len(labels)), predictions] += 1
    acc, f1, kappa, rem_f1, auroc = engine.compute_metrics(predictions, labels, pred_probs=list(probs), testing=True)
    assert abs(acc - 6 / 8) < 1e-9
    from sklearn.metrics import f1_score
    assert abs(rem_f1 - f1_score(labels, predictions, labels=[3], average="macro")) < 1e-9   # REM is class 3
    assert np.isfinite(auroc)
