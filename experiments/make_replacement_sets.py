r"""Sleep-staging datasets for the stage-replacement experiment: real nights with chosen epochs swapped.

  python experiments/make_replacement_sets.py --folds FOLDS/folds.json --real REAL_NIGHTS \
      --generated "NIGHTS/ppg50n1n2_fold{fold}_best/ibi_elgendi_local_fix/night_npz" --output-dir OUT \
      [--arms real gen:Deep gen:REM gen:Deep,REM gen:random0.2 shuffled:Deep shuffled:REM shuffled:random0.2]

REAL_NIGHTS holds every participant's real night NPZ (night_npz_real of a ppg_nights_to_ibi.py run: the
real PPG through the same per-window PPG -> IBI route). --generated is each fold's night_npz, written from
that fold's generator ({fold} is replaced by the fold number); it must cover the fold's training participants.

For every arm and fold, OUT/<arm>/fold<k>/ gets one NPZ per participant of the fold:
  * the classifier's training participants (folds.json "train"): the real night with the chosen epochs replaced;
  * validation and test participants: the real night, untouched (a symbolic link).
Arms (the label never changes, only the 30-s IBI window):
  real                 nothing replaced (the baseline)
  gen:<stages>         every epoch of these stages (W, N1, N2, Deep, REM; comma separated) -> the generated
                       window of the same epoch (generated with that epoch's stage and event conditions)
  gen:random<f>        a random share f of each night's epochs -> generated
  shuffled:<...>       the same epochs, replaced by a real window of the same stage from another training
                       participant of the fold: the control with the same seams but real content
Random epochs are drawn per (fold, participant) from --seed, so gen:random and shuffled:random swap the
same epochs. A replaced NPZ keeps every key and adds `replaced` (1 per replaced 25 Hz sample) and
`replacement` (the arm). OUT/<arm>/fold<k>/manifest.json lists the files and the replaced epochs per stage.
Existing arm folders are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

EPOCH = 750
STAGE_CODES = dict(W=0, N1=1, N2=2, Deep=3, REM=4)
DEFAULT_ARMS = ("real", "gen:Deep", "gen:REM", "gen:Deep,REM", "gen:random0.2", "shuffled:Deep", "shuffled:REM",
                "shuffled:random0.2")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True, help="folder of real night NPZ files (all participants)")
    parser.add_argument("--generated", required=True, help="folder of a fold's generated night NPZ, with {fold}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--only-folds", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20261124)
    return parser.parse_args(argv)


def parse_arm(spec):
    """(name, source, stages or None, fraction or None)."""
    if spec == "real":
        return "real", None, None, None
    source, _, what = spec.partition(":")
    if source not in ("gen", "shuffled") or not what:
        raise ValueError(f"arm {spec!r}: use real, gen:<stages|random<f>> or shuffled:<stages|random<f>>")
    if what.startswith("random"):
        fraction = float(what[len("random"):])
        if not 0 < fraction <= 1:
            raise ValueError(f"arm {spec!r}: the random share must be in (0, 1]")
        return f"{source}_random{round(fraction * 100):g}", source, None, fraction
    stages = what.split(",")
    unknown = [s for s in stages if s not in STAGE_CODES]
    if unknown:
        raise ValueError(f"arm {spec!r}: unknown stages {unknown}; use {sorted(STAGE_CODES)}")
    return f"{source}_{'-'.join(stages)}", source, tuple(stages), None


def epoch_stages(night):
    stages = np.asarray(night["stages"]).reshape(-1)
    if stages.size % EPOCH:
        raise ValueError("night stages are not whole 30-s epochs")
    return stages[::EPOCH].astype(np.int64)


def chosen_epochs(stages, wanted, fraction, seed, fold, sid):
    if wanted is not None:
        return np.flatnonzero(np.isin(stages, [STAGE_CODES[s] for s in wanted]))
    rng = np.random.default_rng([seed, fold, zlib.crc32(sid.encode())])
    count = int(round(fraction * stages.size))
    return np.sort(rng.choice(stages.size, size=count, replace=False)) if count else np.zeros(0, np.int64)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_same_night(real, generated, sid):
    for key in ("stages", "epoch_index"):
        if key in real and key in generated and not np.array_equal(real[key], generated[key]):
            raise ValueError(f"{sid}: generated and real nights differ in {key}")
    if real["data"].shape != generated["data"].shape:
        raise ValueError(f"{sid}: generated and real nights differ in length")


def link(source, target):
    try:
        os.symlink(os.path.abspath(source), target)
    except OSError:
        shutil.copy2(source, target)


def build_fold(fold, arms, real_dir, generated_dir, output, seed):
    sids = sorted(fold["train"] + fold["val"] + fold["test"])
    missing = [s for s in sids if not (real_dir / f"{s}.npz").is_file()]
    if missing:
        raise SystemExit(f"fold {fold['fold']}: no real night for {missing} in {real_dir}")
    real = {s: load(real_dir / f"{s}.npz") for s in fold["train"]}
    stages = {s: epoch_stages(real[s]) for s in fold["train"]}
    generated = {}
    if any(source == "gen" for _, source, _, _ in arms):
        lost = [s for s in fold["train"] if not (generated_dir / f"{s}.npz").is_file()]
        if lost:
            raise SystemExit(f"fold {fold['fold']}: no generated night for {lost} in {generated_dir}")
        for s in fold["train"]:
            generated[s] = load(generated_dir / f"{s}.npz")
            check_same_night(real[s], generated[s], s)
    pool = {}                                     # stage code -> list of (sid, epoch) of real training windows
    for s in fold["train"]:
        for e, code in enumerate(stages[s]):
            pool.setdefault(int(code), []).append((s, e))
    for name, source, wanted, fraction in arms:
        folder = output / name / f"fold{fold['fold']}"
        folder.mkdir(parents=True)
        manifest = dict(arm=name, source=source, stages=wanted, fraction=fraction, fold=fold["fold"], seed=seed,
                        train=fold["train"], val=fold["val"], test=fold["test"], replaced={})
        for s in fold["val"] + fold["test"]:
            link(real_dir / f"{s}.npz", folder / f"{s}.npz")
        for s in fold["train"]:
            if source is None:
                link(real_dir / f"{s}.npz", folder / f"{s}.npz")
                continue
            epochs = chosen_epochs(stages[s], wanted, fraction, seed, fold["fold"], s)
            night = {k: v.copy() for k, v in real[s].items()}
            data = night["data"].reshape(-1, EPOCH)
            mask = np.zeros(data.shape, dtype=np.uint8)
            rng = np.random.default_rng([seed, fold["fold"], zlib.crc32(s.encode()), 1])
            for e in epochs:
                if source == "gen":
                    data[e] = generated[s]["data"].reshape(-1, EPOCH)[e]
                else:
                    candidates = [c for c in pool[int(stages[s][e])] if c[0] != s] or pool[int(stages[s][e])]
                    other, k = candidates[int(rng.integers(len(candidates)))]
                    data[e] = real[other]["data"].reshape(-1, EPOCH)[k]
                mask[e] = 1
            night["data"] = data.reshape(real[s]["data"].shape)
            night["replaced"] = mask.reshape(-1)
            night["replacement"] = np.array(name)
            np.savez_compressed(folder / f"{s}.npz", **night)
            by_stage = {n: int(np.isin(stages[s][epochs], [c]).sum()) for n, c in STAGE_CODES.items()}
            manifest["replaced"][s] = dict(epochs=int(epochs.size), of=int(stages[s].size), by_stage=by_stage)
        total = sum(v["epochs"] for v in manifest["replaced"].values())
        manifest["replaced_share_of_training_epochs"] = (total / sum(stages[s].size for s in fold["train"])
                                                         if fold["train"] else 0.0)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"  {name:22s} fold {fold['fold']}: {total} epochs replaced "
              f"({manifest['replaced_share_of_training_epochs']:.1%} of the training epochs)")


def main(argv=None):
    args = parse_args(argv)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    arms = [parse_arm(a) for a in args.arms]
    names = [a[0] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"repeated arms: {names}")
    output = args.output_dir.expanduser().resolve()
    taken = [n for n in names if (output / n).exists()]
    if taken:
        raise SystemExit(f"arm folders exist already, never overwritten: {taken} in {output}")
    real_dir = args.real.expanduser().resolve()
    for fold in folds:
        if args.only_folds is not None and fold["fold"] not in args.only_folds:
            continue
        generated_dir = Path(args.generated.format(fold=fold["fold"])).expanduser().resolve()
        build_fold(fold, arms, real_dir, generated_dir, output, args.seed)
    log = output / "arms.json"                                    # one entry per invocation
    runs = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else []
    runs.append(dict(folds=str(args.folds.resolve()), real=str(real_dir), generated=args.generated, arms=args.arms,
                     names=names, seed=args.seed, only_folds=args.only_folds))
    log.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print(f"output: {output}")
    return names


if __name__ == "__main__":
    main()
