r"""Participant-level folds for the stage-replacement experiment.

  python experiments/make_folds.py --nights REAL_NIGHTS --output FOLDS/folds.json [--folds 5] [--seed 20261124]
      [--metadata participant_info.csv]

REAL_NIGHTS holds one real night NPZ per participant (night_npz_real of dataset-generation-ppg50's
ppg_nights_to_ibi.py: `data`, `stages` per 25 Hz sample, `fs`, `ahi`). Participants are split into K folds
balanced on deep sleep (at least --deep-epochs N3 epochs, default 10 = 5 min) and on AHI: within each of the
two groups, participants sorted by AHI are dealt in blocks of K, each block in random order to the folds with
the fewest members. In every fold, --val-fraction of all participants (default 0.1) are taken from the
training side, the same way, as the classifier's validation participants.

folds.json, per fold: test (always real), val and train (the classifier), generator_train = train + val (the
participants a fold's generator may be trained on: never the fold's test participants); per participant: AHI,
deep epochs, epochs, test fold. AHI comes from the NPZ, or from --metadata (participant_info.csv) when the
NPZ has none. An existing folds.json is never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEEP_CODE = 3                               # stage codes in the night NPZ: W 0, N1 1, N2 2, N3 3, REM 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nights", type=Path, required=True, help="folder of real night NPZ files")
    parser.add_argument("--output", type=Path, required=True, help="folds.json to write")
    parser.add_argument("--metadata", type=Path, default=None, help="participant_info.csv (SID, AHI) if needed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--deep-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261124)
    return parser.parse_args(argv)


def participants(nights, metadata=None):
    """{sid: dict(ahi, deep_epochs, epochs)} from the night NPZ files."""
    table = {}
    if metadata is not None:
        import csv

        with Path(metadata).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    table[row["SID"].strip()] = float(row["AHI"])
                except (KeyError, ValueError):
                    pass
    out = {}
    for path in sorted(Path(nights).glob("S*.npz")):
        with np.load(path) as z:
            fs = int(np.asarray(z["fs"]).item())
            stages = np.asarray(z["stages"]).reshape(-1)[:: fs * 30]
            ahi = float(np.asarray(z["ahi"]).item()) if "ahi" in z.files else float("nan")
        if not np.isfinite(ahi):
            ahi = table.get(path.stem, float("nan"))
        out[path.stem] = dict(ahi=ahi, deep_epochs=int((stages == DEEP_CODE).sum()), epochs=int(stages.size))
    return out


def deal(sids, people, k, rng, sizes, deep_epochs):
    """Assign sids to k bins: deep-sleep groups, AHI order, blocks of k dealt to the smallest bins."""
    assignment = {}
    key = lambda s: (not np.isfinite(people[s]["ahi"]), people[s]["ahi"] if np.isfinite(people[s]["ahi"]) else 0, s)
    for group in (True, False):
        members = sorted((s for s in sids if (people[s]["deep_epochs"] >= deep_epochs) == group), key=key)
        for start in range(0, len(members), k):
            block = list(rng.permutation(members[start:start + k]))
            order = sorted(range(k), key=lambda b: (sizes[b], rng.random()))
            for sid, b in zip(block, order):
                assignment[sid] = b
                sizes[b] += 1
    return assignment


def make_folds(people, k, val_fraction, deep_epochs, seed):
    rng = np.random.default_rng(seed)
    sids = sorted(people)
    if len(sids) < 2 * k:
        raise ValueError(f"{len(sids)} participants are too few for {k} folds")
    test_of = deal(sids, people, k, rng, [0] * k, deep_epochs)
    n_val = max(1, int(round(val_fraction * len(sids))))
    folds = []
    for fold in range(k):
        test = sorted(s for s in sids if test_of[s] == fold)
        rest = [s for s in sids if test_of[s] != fold]
        bins = max(2, int(round(len(rest) / n_val)))
        val_of = deal(rest, people, bins, rng, [0] * bins, deep_epochs)
        val = sorted(s for s in rest if val_of[s] == 0)
        train = sorted(s for s in rest if val_of[s] != 0)
        folds.append(dict(fold=fold, test=test, val=val, train=train, generator_train=sorted(train + val)))
    return folds, test_of


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"{output} exists; folds are never overwritten (give another --output)")
    people = participants(args.nights.expanduser().resolve(), args.metadata)
    folds, test_of = make_folds(people, args.folds, args.val_fraction, args.deep_epochs, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(schema="ibi_folds_v1", folds_count=args.folds, seed=args.seed, val_fraction=args.val_fraction,
                   deep_epochs_threshold=args.deep_epochs, nights=str(args.nights.expanduser().resolve()),
                   folds=folds, participants={s: dict(people[s], fold=test_of[s]) for s in sorted(people)})
    output.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(people)} participants -> {args.folds} folds in {output}")
    for fold in folds:
        deep = sum(people[s]["deep_epochs"] >= args.deep_epochs for s in fold["test"])
        ahi = np.nanmedian([people[s]["ahi"] for s in fold["test"]])
        print(f"  fold {fold['fold']}: test {len(fold['test'])} (deep sleep {deep}, median AHI {ahi:.1f}), "
              f"val {len(fold['val'])}, train {len(fold['train'])}")
    return summary


if __name__ == "__main__":
    main()
