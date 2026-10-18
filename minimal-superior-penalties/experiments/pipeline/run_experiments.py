#!/usr/bin/env python3
"""Orchestrator: train once per (dataset, regime, seed), score every lambda
grid point and baseline offline against the cached epoch stream, write
results/*.csv. Matches experiments/config.yaml's compute_plan exactly.

Usage:
    python -m experiments.pipeline.run_experiments [--smoke] [--datasets ...]
                                                     [--regimes ...] [--seeds ...]
                                                     [--epochs N] [--train-frac F]
                                                     [--resume]

--smoke runs a fast, tiny-scale sanity pass (small data subset, 2 epochs,
1 seed) to validate the pipeline end-to-end. It writes to
experiments/smoke_output/, never to results/ -- results/ is frozen output
for the real pre-registered runs only (config.yaml: `frozen: true, never
overwrite`).

--epochs / --train-frac override the config's per-regime values -- for
calibration timing runs, not for the pre-registered sweep itself.

Every cell's rows are written incrementally (appended to the CSVs) as soon
as that cell finishes, and a `.done` marker is left in its run directory
*last*, only after every output file has been written -- so the marker's
presence is a reliable signal that the cell is fully done. --resume skips
any (dataset, regime, seed) cell whose marker is already present.

A cell that lacks a marker is purged from every output CSV before being
(re)trained (IncrementalWriter.purge_cell), not just retrained -- an
interruption between "rows written" and "marker written" (Colab
disconnects hit exactly this window) used to leave an orphaned, incomplete
attempt's rows sitting alongside the next attempt's, silently doubling that
cell's row count. Purging first makes a retry idempotent: at most one
attempt's rows are ever present for a given cell, however many times it
gets interrupted and resumed.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from .config import load_config, config_hash, git_commit
from .data import build_splits
from .train import train_one_run, score_selected_epoch
from .selection import select
from . import scoring as sc

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "experiments" / "torchvision_data"


def brier_lambdas(cfg, classes):
    # AMENDMENT 1: the grid is keyed by class
    # count, since lambda* and every other grid point are c-dependent.
    return [(g["name"], g["value"])
            for g in cfg["lambda_grid"]["by_classes"][classes]["brier"]]


def log_lambdas(cfg, classes):
    return [(g["name"], g["value"])
            for g in cfg["lambda_grid"]["by_classes"][classes]["log"]]


def run_cell(cfg, dataset_cfg, regime_cfg, seed, run_root, device,
             epochs_override=None, train_frac_override=None,
             batch_size_override=None):
    """Trains one (dataset, regime, seed) cell and scores every lambda grid
    point and baseline, both selection modes, returning
    (metric_rows, diagnostics_row, elapsed_seconds).
    """
    did, rid = dataset_cfg["id"], regime_cfg["id"]
    run_dir = run_root / f"{did}_{rid}_seed{seed}"
    print(f"  [{did}/{rid}/seed={seed}] building splits...", flush=True)

    effective_regime = dict(regime_cfg)
    if train_frac_override is not None:
        effective_regime["train_fraction"] = train_frac_override

    train_ds, val_ds, test_ds = build_splits(dataset_cfg, effective_regime, seed, DATA_ROOT)
    print(f"  [{did}/{rid}/seed={seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    t0 = time.time()
    epoch_cache, _, (in_ch, classes, hw, backbone_name) = train_one_run(
        dataset_cfg, effective_regime, cfg["backbones"], cfg["model"], seed,
        train_ds, val_ds, test_ds, run_dir, device=device,
        epochs_override=epochs_override, batch_size_override=batch_size_override,
    )
    train_elapsed = time.time() - t0
    print(f"  [{did}/{rid}/seed={seed}] trained {len(epoch_cache)} epochs in {train_elapsed:.1f}s"
          f" ({train_elapsed/max(len(epoch_cache),1):.1f}s/epoch)", flush=True)

    rows = []
    modes = cfg["selection"]["modes"]
    patience = cfg["selection"]["early_stopping_patience"]
    min_delta = cfg["selection"]["min_delta"]
    restore_best = cfg["selection"]["restore_best"]

    def selection_criteria_for(fn):
        return {e: fn(d["val_probs"], d["val_labels"]) for e, d in epoch_cache.items()}

    test_cache = {}

    def test_metrics_at(epoch):
        if epoch not in test_cache:
            probs, labels = score_selected_epoch(
                epoch, backbone_name, dataset_cfg, cfg["backbones"], test_ds, run_dir, device
            )
            test_cache[epoch] = (probs, labels)
        return test_cache[epoch]

    candidates = []
    for name, lam in brier_lambdas(cfg, classes):
        candidates.append(("brier", name, lam, lambda p, l, lam=lam: sc.s_lambda_score(p, l, lam, "brier")))
    for name, lam in log_lambdas(cfg, classes):
        candidates.append(("log", name, lam, lambda p, l, lam=lam: sc.s_lambda_score(p, l, lam, "log")))
    candidates.append(("baseline", "monitor_macro_f1_direct", None,
                        lambda p, l: -sc.macro_f1(p, l, classes)))
    candidates.append(("baseline", "monitor_error_rate_direct", None,
                        lambda p, l: sc.error_rate(p, l)))
    for w in cfg["baselines"][2]["weights"]:
        candidates.append(("baseline", f"weighted_mix_{w}", w,
                            lambda p, l, w=w: sc.weighted_mixture_score(p, l, w)))

    for base, lam_name, lam_val, crit_fn in candidates:
        crit_by_epoch = selection_criteria_for(crit_fn)
        for mode in modes:
            sel_epoch = select(mode, crit_by_epoch, patience, min_delta, restore_best)
            test_probs, test_labels = test_metrics_at(sel_epoch)
            common = dict(
                run_id=f"{did}_{rid}_seed{seed}_{base}_{lam_name}_{mode}",
                dataset=did, classes=classes, regime=rid, base=base,
                lambda_name=lam_name, lambda_value=lam_val, selection_mode=mode,
                seed=seed, config_hash=config_hash(cfg), git_commit=git_commit(),
                selected_epoch=sel_epoch,
            )
            rows.append(dict(common, metric="macro_f1",
                              value=sc.macro_f1(test_probs, test_labels, classes)))
            rows.append(dict(common, metric="ece",
                              value=sc.ece(test_probs, test_labels,
                                           bins=cfg["outcomes"][1]["bins"])))
            rows.append(dict(common, metric="nll", value=sc.nll(test_probs, test_labels)))

    star_crit = selection_criteria_for(lambda p, l: sc.s_lambda_score(p, l, dict(brier_lambdas(cfg, classes))["star"], "brier"))
    star_epoch = select("checkpointing", star_crit)
    val_at_star = epoch_cache[star_epoch]
    diag = sc.diagnostics(val_at_star["val_probs"], val_at_star["val_labels"])
    diag_row = dict(dataset=did, regime=rid, seed=seed,
                     config_hash=config_hash(cfg), git_commit=git_commit(),
                     achieved_macro_f1=sc.macro_f1(val_at_star["val_probs"], val_at_star["val_labels"], classes),
                     **diag)

    return rows, diag_row, time.time() - t0


class IncrementalWriter:
    """Appends rows to a CSV as soon as they're available, instead of
    holding everything in memory until the very end -- so an interrupted
    multi-day run has already saved every cell that finished.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header = None
        if self.path.exists():
            with open(self.path, "r", newline="", encoding="utf-8") as fh:
                r = csv.reader(fh)
                self._header = next(r, None)

    def append(self, rows):
        if not rows:
            return
        if self._header is None:
            self._header = sorted({k for r in rows for k in r})
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self._header).writeheader()
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self._header)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in self._header})

    def purge_cell(self, did, rid, seed):
        """Removes any existing rows for this (dataset, regime, seed) cell.

        Called before (re)processing a cell that lacks a .done marker, so a
        retried cell never leaves an earlier, interrupted attempt's rows
        sitting alongside the new ones -- see the module docstring. No-op if
        the file doesn't exist yet or has no matching rows (the common
        case: most cells are never actually retried).
        """
        if not self.path.exists():
            return
        with open(self.path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        kept = [r for r in rows
                if not (r.get("dataset") == did and r.get("regime") == rid
                        and str(r.get("seed")) == str(seed))]
        if len(kept) == len(rows):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept)


def cell_done_marker(run_root, did, rid, seed):
    return run_root / f"{did}_{rid}_seed{seed}" / ".done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="tiny fast sanity pass, writes to experiments/smoke_output/")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                     help="override every regime's epoch count (calibration runs)")
    ap.add_argument("--train-frac", type=float, default=None,
                     help="override train_fraction (calibration runs)")
    ap.add_argument("--resume", action="store_true",
                     help="skip cells whose .done marker already exists")
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  config_hash={config_hash(cfg)}  git_commit={git_commit()}", flush=True)

    datasets = cfg["datasets"]
    regimes = cfg["regimes"]
    seeds = cfg["seeds"]
    if args.datasets:
        datasets = [d for d in datasets if d["id"] in args.datasets]
    if args.regimes:
        regimes = [r for r in regimes if r["id"] in args.regimes]
    if args.seeds:
        seeds = args.seeds

    epochs_override, train_frac_override, batch_override = args.epochs, args.train_frac, None

    if args.smoke:
        out_dir = ROOT / "experiments" / "smoke_output"
        run_root = out_dir / "runs"
        epochs_override, train_frac_override, batch_override = (
            args.epochs or 2, args.train_frac if args.train_frac is not None else 0.02, 64)
        print("SMOKE MODE: validating the pipeline, not producing results", flush=True)
    else:
        is_calibration = epochs_override is not None or train_frac_override is not None
        if is_calibration:
            # Calibration/timing runs NEVER touch results/ or the real
            # experiments/runs/ checkpoints -- separate directory entirely,
            # so a timing probe can't seed the frozen output with a
            # non-representative single-epoch number.
            out_dir = ROOT / "experiments" / "calibration_output"
            run_root = out_dir / "runs"
            print("CALIBRATION MODE: writing to experiments/calibration_output/, "
                  "not results/", flush=True)
        else:
            out_dir = ROOT / cfg["outputs"]["dir"]
            run_root = ROOT / "experiments" / "runs"
            if not args.resume and out_dir.exists() and any(out_dir.glob("*.csv")):
                raise SystemExit(
                    f"{out_dir} already has result CSVs and outputs.frozen=true in config.yaml. "
                    "Refusing to overwrite -- pass --resume to continue an interrupted run, "
                    "or archive/rename the old results first."
                )

    files = cfg["outputs"]["files"]
    writers = {
        "macro_f1": IncrementalWriter(out_dir / files["f1"]),
        "ece": IncrementalWriter(out_dir / files["ece"]),
        "nll": IncrementalWriter(out_dir / files["nll"]),
    }
    margins_writer = IncrementalWriter(out_dir / files["margins"])
    raw_writer = IncrementalWriter(out_dir / "raw_all_metrics.csv")

    cells = [(d, r, s) for d in datasets for r in regimes for s in seeds]
    total = len(cells)
    done_count = 0
    t_start = time.time()

    for i, (dataset_cfg, regime_cfg, seed) in enumerate(cells, 1):
        did, rid = dataset_cfg["id"], regime_cfg["id"]
        marker = cell_done_marker(run_root, did, rid, seed)
        if args.resume and marker.exists():
            print(f"[{i}/{total}] {did}/{rid}/seed={seed}: already done, skipping", flush=True)
            done_count += 1
            continue

        for writer in (*writers.values(), margins_writer, raw_writer):
            writer.purge_cell(did, rid, seed)

        print(f"[{i}/{total}] {did}/{rid}/seed={seed}: starting "
              f"(elapsed so far: {time.time()-t_start:.0f}s)", flush=True)
        rows, diag, elapsed = run_cell(cfg, dataset_cfg, regime_cfg, seed, run_root, device,
                                        epochs_override, train_frac_override, batch_override)

        for metric, writer in writers.items():
            writer.append([r for r in rows if r["metric"] == metric])
        margins_writer.append([diag])
        raw_writer.append(rows)

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"completed in {elapsed:.1f}s\n")
        done_count += 1
        avg = (time.time() - t_start) / done_count
        remaining = total - done_count
        print(f"[{i}/{total}] {did}/{rid}/seed={seed}: done in {elapsed:.1f}s. "
              f"{remaining} cells left, ~{avg*remaining/3600:.1f}h remaining at current average pace",
              flush=True)

    print(f"all {total} cells complete, wrote to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
