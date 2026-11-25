r"""Train and test a sleep-staging model on our fixed participant folds: DREAMT only, from scratch, per arm and seed.

  python experiments/run_folds.py --model mambasleepnet --folds FOLDS/folds.json --sets SETS --output-dir RESULTS \
      [--arms real gen_Deep ...] [--seeds 0 1 2] [--epochs N] [--patience N]

--model is one of watchsleepnet, insightsleepnet, sleepconvnet, sleepppgnet, mambasleepnet; its settings come
from modeling/config.py (<Model>Config). MambaSleepNet is the CNN + bidirectional Mamba with only a classifier
(no contrastive learning) and needs mamba_ssm with CUDA. Every model trains with cross-entropy.

SETS is the output of experiments/make_replacement_sets.py: SETS/<arm>/fold<k>/<SID>.npz, where the training
participants carry the arm's replaced windows and validation and test participants are real. For every arm
and seed, the five folds go through this repository's engine.train_cross_validate (its optimizer, early
stopping on validation kappa and loss, checkpoints) with our folds instead of a random KFold. Labels are those
of data_setup.SSDataset: W, Light, Deep, REM. Then every fold's best checkpoint predicts each test
participant, and we score:
  per fold and pooled over the five test folds: accuracy, Cohen's kappa, macro F1, F1 per class, confusion
  matrix; per test participant: accuracy and kappa.
Output: RESULTS/<model>/<arm>/seed<s>/ (checkpoints, results.json, per_fold.csv, per_participant.csv) and,
over all finished runs of the model, RESULTS/<model>/summary.csv and summary.json: mean and SD over seeds of
the pooled metrics and the difference to the "real" arm per seed (same seeds, same folds). A finished run is
not repeated. --check loads one batch of every loader of the first arm and runs the model on it (no training).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

TASK = "sleep_staging"
DATASET = "dreamt_pibi"
ROOT = Path(__file__).resolve().parents[1]
MODELS = dict(watchsleepnet="WatchSleepNetConfig", insightsleepnet="InsightSleepNetConfig",
              sleepconvnet="SleepConvNetConfig", sleepppgnet="SleepPPGNetConfig", mambasleepnet="MambaSleepNetConfig")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--sets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=None, help="default: every arm folder in --sets")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=None, help="default: the config's NUM_EPOCHS")
    parser.add_argument("--patience", type=int, default=None, help="default: the config's PATIENCE")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-params", default=None, help="JSON overriding model constructor fields (tests)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="hide the engine's own printing")
    return parser.parse_args(argv)


def import_modeling():
    modeling = ROOT / "modeling"
    if str(modeling) not in sys.path:
        sys.path.insert(0, str(modeling))
    import config
    import data_setup
    import engine
    return config, data_setup, engine


def class_names(data_setup):
    codes = np.array([0, 1, 2, 3, 4])
    mapped = data_setup.SSDataset.remap_labels(object.__new__(data_setup.SSDataset), codes, TASK)
    count = int(np.max(mapped)) + 1
    names = {}
    for code, label in zip(("W", "N1", "N2", "N3", "REM"), mapped):
        names.setdefault(int(label), []).append(code)
    readable = {("W",): "W", ("N1", "N2"): "Light", ("N3",): "Deep", ("REM",): "REM", ("N1", "N2", "N3"): "NREM"}
    return [readable.get(tuple(names.get(k, [])), "+".join(names.get(k, [str(k)]))) for k in range(count)]


def set_seed(seed):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(data_setup, folder, sids, batch_size, shuffle, workers):
    from torch.utils.data import DataLoader

    dataset = data_setup.SSDataset(dir=Path(folder), dataset=DATASET, file_list=[f"{s}.npz" for s in sids], task=TASK)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=data_setup.SSDataset.collate_fn)


def scores(true, pred, n_classes):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

    true, pred = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    keep = true >= 0
    true, pred = true[keep], pred[keep]
    labels = list(range(n_classes))
    if true.size == 0:
        return dict(epochs=0)
    return dict(epochs=int(true.size), accuracy=float(accuracy_score(true, pred)),
                kappa=float(cohen_kappa_score(true, pred, labels=labels)),
                macro_f1=float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
                f1=[float(v) for v in f1_score(true, pred, labels=labels, average=None, zero_division=0)],
                confusion=confusion_matrix(true, pred, labels=labels).tolist())


def run_one(arm, seed, folds, args, modules, names, model_params):
    import torch

    config, data_setup, engine = modules
    output = args.output_dir / args.model / arm / f"seed{seed}"
    if (output / "results.json").is_file():
        print(f"{arm} seed {seed}: finished before, kept")
        return json.loads((output / "results.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = args.batch_size or model_params.get("BATCH_SIZE", 16)
    set_seed(seed)
    folders = [args.sets / arm / f"fold{f['fold']}" for f in folds]
    loaders = [(loader(data_setup, d, f["train"], batch, True, args.num_workers),
                loader(data_setup, d, f["val"], batch, False, args.num_workers),
                loader(data_setup, d, f["test"], batch, False, args.num_workers)) for d, f in zip(folders, folds)]
    lr = model_params.get("LEARNING_RATE", 1e-3)
    wd = model_params.get("WEIGHT_DECAY", 1e-4)
    loss_fn = model_params.get("LOSS_FN") or torch.nn.CrossEntropyLoss(ignore_index=-1)
    checkpoint = str(output / f"{args.model}.pt")
    quiet = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
    with quiet:
        engine.train_cross_validate(model_name=args.model, model_params=model_params, dataloader_folds=loaders,
                                    saved_model_path=None, learning_rate=lr, weight_decay=wd, loss_fn=loss_fn,
                                    num_epochs=args.epochs or model_params.get("NUM_EPOCHS", 100),
                                    patience=args.patience or model_params.get("PATIENCE", 10), device=device,
                                    checkpoint_path=checkpoint, freeze_layers=False)
    per_fold, per_participant, pooled_true, pooled_pred = [], [], [], []
    for k, (fold, folder) in enumerate(zip(folds, folders), start=1):
        path = checkpoint.replace(".pt", f"_fold{k}.pt")
        model, _ = engine.setup_model_and_optimizer(model_name=args.model, model_params=model_params,
                                                    device=device, saved_model_path=None, learning_rate=lr,
                                                    weight_decay=wd, freeze_layers=False)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        fold_true, fold_pred = [], []
        for sid in fold["test"]:
            true, pred, _, _ = engine.test_step(model, loader(data_setup, folder, [sid], 1, False, 0), device, task=TASK)
            s = scores(true, pred, len(names))
            per_participant.append(dict(fold=fold["fold"], subject=sid, epochs=s["epochs"], accuracy=s.get("accuracy"),
                                        kappa=s.get("kappa")))
            fold_true.extend(true)
            fold_pred.extend(pred)
        per_fold.append(dict(fold=fold["fold"], checkpoint_found=os.path.exists(path),
                             **scores(fold_true, fold_pred, len(names))))
        pooled_true.extend(fold_true)
        pooled_pred.extend(fold_pred)
    result = dict(model=args.model, arm=arm, seed=seed, classes=names,
                  pooled=scores(pooled_true, pooled_pred, len(names)),
                  per_fold=per_fold, settings=dict(batch_size=batch, learning_rate=lr, weight_decay=wd,
                  epochs=args.epochs or model_params.get("NUM_EPOCHS"), patience=args.patience or model_params.get("PATIENCE"),
                  model={k: v for k, v in model_params.items() if isinstance(v, (int, float, str, bool))}))
    with (output / "per_fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "epochs", "accuracy", "kappa", "macro_f1"] + [f"f1_{n}" for n in names])
        for row in per_fold:
            writer.writerow([row["fold"], row["epochs"], row.get("accuracy"), row.get("kappa"), row.get("macro_f1")]
                            + list(row.get("f1", [None] * len(names))))
    with (output / "per_participant.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "subject", "epochs", "accuracy", "kappa"])
        writer.writeheader()
        writer.writerows(per_participant)
    (output / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    p = result["pooled"]
    print(f"{args.model} {arm} seed {seed}: accuracy {p['accuracy']:.3f}, kappa {p['kappa']:.3f}, macro F1 {p['macro_f1']:.3f}, "
          + ", ".join(f"F1 {n} {v:.3f}" for n, v in zip(names, p["f1"])))
    return result


def summarize(output_dir, names):
    runs = {}
    for path in sorted(output_dir.glob("*/seed*/results.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    metrics = ["accuracy", "kappa", "macro_f1"] + [f"f1_{n}" for n in names]

    def value(r, m):
        p = r["pooled"]
        return p["f1"][names.index(m[3:])] if m.startswith("f1_") else p[m]

    rows, summary = [], {}
    for arm, seeds in runs.items():
        entry = dict(seeds=sorted(seeds))
        for m in metrics:
            values = np.array([value(seeds[s], m) for s in sorted(seeds)])
            entry[m] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if values.size > 1 else 0.0)
            if arm != "real" and "real" in runs:
                common = sorted(set(seeds) & set(runs["real"]))
                if common:
                    diff = np.array([value(seeds[s], m) - value(runs["real"][s], m) for s in common])
                    entry[m].update(diff_vs_real_mean=float(diff.mean()),
                                    diff_vs_real_sd=float(diff.std(ddof=1)) if diff.size > 1 else 0.0)
        summary[arm] = entry
        rows.append([arm, len(seeds)] + [f"{entry[m]['mean']:.4f} ± {entry[m]['sd']:.4f}" for m in metrics]
                    + [f"{entry[m].get('diff_vs_real_mean', float('nan')):+.4f}" for m in metrics])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "seeds"] + metrics + [f"{m}_minus_real" for m in metrics])
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(dict(classes=names, arms=summary), indent=1), encoding="utf-8")
    return summary


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.sets = args.sets.expanduser().resolve()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    arms = args.arms or sorted(p.name for p in args.sets.iterdir() if p.is_dir() and (p / "fold0").is_dir())
    missing = [f"{a}/fold{f['fold']}" for a in arms for f in folds if not (args.sets / a / f"fold{f['fold']}").is_dir()]
    if missing:
        raise SystemExit(f"missing in {args.sets}: {missing}")
    modules = import_modeling()
    config, data_setup, _ = modules
    names = class_names(data_setup)
    model_params = getattr(config, MODELS[args.model]).to_dict()
    if model_params.get("num_classes") != len(names):
        print(f"note: {MODELS[args.model]} has num_classes {model_params.get('num_classes')}, the labels give "
              f"{len(names)} ({', '.join(names)}); using {len(names)}")
        model_params["num_classes"] = len(names)
    if "output_size" in model_params:
        model_params["output_size"] = len(names)
    if args.model_params:
        model_params.update(json.loads(args.model_params))
    if args.check:
        batch = args.batch_size or model_params.get("BATCH_SIZE", 16)
        for f in folds:
            folder = args.sets / arms[0] / f"fold{f['fold']}"
            for part in ("train", "val", "test"):
                x, y, lengths, _ = next(iter(loader(data_setup, folder, f[part], batch, False, 0)))
                print(f"{arms[0]} fold {f['fold']} {part}: {len(f[part])} nights, batch x {tuple(x.shape)}, "
                      f"labels {sorted(set(y[y >= 0].tolist()))}")
        print(f"classes: {names}; arms: {arms}")
        return dict(status="checked", classes=names, arms=arms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for seed in args.seeds:
            run_one(arm, seed, folds, args, modules, names, model_params)
    summary = summarize(args.output_dir / args.model, names)
    print(f"summary: {args.output_dir / args.model / 'summary.csv'}")
    return summary


if __name__ == "__main__":
    main()
