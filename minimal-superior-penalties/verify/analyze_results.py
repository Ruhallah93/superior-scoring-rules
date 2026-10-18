"""Aggregates results/*.csv into every number results.md needs.

Mirrors figures/make_figures.py's discipline: nothing in the manuscript is a
hand-typed or eyeballed number. This script reads the frozen, pipeline-only
CSVs in results/ (never writes there -- see the repo map) and writes a single
JSON report to verify/results_report.json. Every number in results.md is
copied from that file, not recomputed in prose.

Run: python verify/analyze_results.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_PATH = ROOT / "verify" / "results_report.json"

CONFIG = yaml.safe_load(open(ROOT / "experiments" / "config.yaml", encoding="utf-8"))
INFORMATIVE_THRESH = CONFIG["regime_reporting"]["informative_threshold_macro_f1"]
ALPHA = CONFIG["decision_rule"]["alpha"]

# The pre-registered design's ACTUAL scope. config.yaml's `datasets` x
# `regimes` is a flat cross-product and does not by itself encode that an
# amendment can scope a dataset to fewer regimes than the others (Amendment
# 1: emnist_balanced, full regime only).
# Anything in results/*.csv outside this explicit list is NOT part of the
# design -- e.g. a single orphaned cifar100/full/seed=0 cell was found in
# the wild here once, left over from a config version that was amended away
# before any run was supposed to happen against it. Such rows are reported
# and excluded, never silently aggregated in.
DATASET_REGIME_SCOPE = {
    "mnist": ["full", "subset", "noisy", "hard"],
    "cifar10": ["full", "subset", "noisy", "hard"],
    "emnist_balanced": ["full"],
}


def load(fn):
    with open(RESULTS / fn, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def holm(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the ORIGINAL input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def main():
    f1_rows = load("f1.csv")
    ece_rows = load("ece.csv")
    nll_rows = load("nll.csv")
    margins_rows = load("margins.csv")

    seeds = CONFIG["seeds"]
    expected_cells = {(d, r, s) for d, regs in DATASET_REGIME_SCOPE.items()
                       for r in regs for s in seeds}

    def cell_of(row):
        return (row["dataset"], row["regime"], int(row["seed"]))

    # Split every file into in-scope and out-of-scope rows. Out-of-scope
    # rows (a cell not in DATASET_REGIME_SCOPE, e.g. an orphaned pilot run
    # from an amended-away config) are reported, never aggregated.
    excluded_report = {}
    named = [("f1", f1_rows), ("ece", ece_rows), ("nll", nll_rows), ("margins", margins_rows)]
    in_scope = {}
    for name, rows in named:
        keep = [r for r in rows if cell_of(r) in expected_cells]
        drop = [r for r in rows if cell_of(r) not in expected_cells]
        in_scope[name] = keep
        if drop:
            excluded_report[name] = sorted({cell_of(r) for r in drop})
    f1_rows, ece_rows, nll_rows, margins_rows = (
        in_scope["f1"], in_scope["ece"], in_scope["nll"], in_scope["margins"])

    # completeness check on the in-scope rows only -- fail loudly rather
    # than silently report on a partial grid
    for name, rows in [("f1", f1_rows), ("ece", ece_rows), ("nll", nll_rows), ("margins", margins_rows)]:
        seen = {cell_of(r) for r in rows}
        missing = expected_cells - seen
        assert not missing, f"{name}.csv missing cells: {sorted(missing)}"
        counts = defaultdict(int)
        for r in rows:
            counts[cell_of(r)] += 1
        expected_per_cell = 1 if name == "margins" else 48
        bad = {k: v for k, v in counts.items() if v != expected_per_cell}
        assert not bad, f"{name}.csv has cells with unexpected row count: {bad}"

    # config_hash: different cells legitimately carry different hashes
    # (each amendment changes config.yaml's bytes) -- what must NOT happen
    # is a single CELL split across two hashes, which would mean its rows
    # came from two different config versions. Record one hash per cell.
    all_in_scope = f1_rows + ece_rows + nll_rows + margins_rows
    hash_by_cell = defaultdict(set)
    commit_by_cell = defaultdict(set)
    for r in all_in_scope:
        hash_by_cell[cell_of(r)].add(r["config_hash"])
        commit_by_cell[cell_of(r)].add(r["git_commit"])
    split_cells = {k: v for k, v in hash_by_cell.items() if len(v) > 1}
    assert not split_cells, f"cells spanning more than one config_hash: {split_cells}"

    hash_of_cell = {k: next(iter(v)) for k, v in hash_by_cell.items()}
    hashes_present = sorted(set(hash_of_cell.values()))
    provenance = {
        "config_hashes_present": hashes_present,
        "config_hash_by_cell": {f"{d}/{r}/seed{s}": h for (d, r, s), h in
                                  sorted(hash_of_cell.items())},
        "git_commit": next(iter({next(iter(v)) for v in commit_by_cell.values()})),
        "n_f1_rows": len(f1_rows), "n_ece_rows": len(ece_rows),
        "n_nll_rows": len(nll_rows), "n_margins_rows": len(margins_rows),
        "excluded_out_of_scope_cells": excluded_report,
    }

    # ---------------------------------------------------- regime informativeness
    regime_f1 = defaultdict(list)
    for r in margins_rows:
        regime_f1[(r["dataset"], r["regime"])].append(float(r["achieved_macro_f1"]))

    informativeness = {}
    for (d, r), vals in regime_f1.items():
        mean_f1 = float(np.mean(vals))
        informativeness[f"{d}/{r}"] = {
            "mean_achieved_macro_f1": mean_f1,
            "std_achieved_macro_f1": float(np.std(vals, ddof=1)),
            "n_seeds": len(vals),
            "informative": bool(mean_f1 <= INFORMATIVE_THRESH),
        }

    # ---------------------------------------------------- diagnostics per regime
    diag_fields = ["margin_mean", "margin_median", "near_tie_rate", "penalty_fire_rate"]
    diagnostics = {}
    per_regime_diag = defaultdict(lambda: defaultdict(list))
    for r in margins_rows:
        key = f"{r['dataset']}/{r['regime']}"
        for f in diag_fields:
            per_regime_diag[key][f].append(float(r[f]))
    for key, fields in per_regime_diag.items():
        diagnostics[key] = {f: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                             for f, v in fields.items()}

    # ---------------------------------------------------- frontier: mean/std per
    # (dataset, regime, base, lambda_name, lambda_value, selection_mode), across seeds
    def build_frontier(rows, metric_name):
        groups = defaultdict(list)
        for r in rows:
            key = (r["dataset"], r["regime"], r["base"], r["lambda_name"],
                   r["lambda_value"], r["selection_mode"])
            groups[key].append(float(r["value"]))
        out = []
        for (d, rg, base, lam_name, lam_val, mode), vals in groups.items():
            out.append({
                "dataset": d, "regime": rg, "base": base, "lambda_name": lam_name,
                "lambda_value": (float(lam_val) if lam_val not in ("", None) else None),
                "selection_mode": mode, "metric": metric_name,
                "n_seeds": len(vals), "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })
        return out

    frontier = (build_frontier(f1_rows, "macro_f1")
                + build_frontier(ece_rows, "ece")
                + build_frontier(nll_rows, "nll"))

    # ---------------------------------------------------- stability: std/IQR of
    # selected_epoch across seeds, per (dataset, regime, base, lambda_name, mode)
    ep_groups = defaultdict(list)
    for r in f1_rows:  # selected_epoch is identical across f1/ece/nll for a given run_id
        key = (r["dataset"], r["regime"], r["base"], r["lambda_name"], r["selection_mode"])
        ep_groups[key].append(int(r["selected_epoch"]))
    stability = []
    for (d, rg, base, lam_name, mode), eps in ep_groups.items():
        arr = np.array(eps, dtype=float)
        q75, q25 = np.percentile(arr, [75, 25])
        stability.append({
            "dataset": d, "regime": rg, "base": base, "lambda_name": lam_name,
            "selection_mode": mode, "n_seeds": len(eps),
            "epochs": eps, "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "iqr": float(q75 - q25),
        })

    # ---------------------------------------------------- baselines
    baseline_rows = {
        "macro_f1": [r for r in f1_rows if r["base"] == "baseline"],
        "ece": [r for r in ece_rows if r["base"] == "baseline"],
        "nll": [r for r in nll_rows if r["base"] == "baseline"],
    }
    baselines = []
    for metric, rows in baseline_rows.items():
        groups = defaultdict(list)
        for r in rows:
            key = (r["dataset"], r["regime"], r["lambda_name"], r["selection_mode"])
            groups[key].append(float(r["value"]))
        for (d, rg, name, mode), vals in groups.items():
            baselines.append({
                "dataset": d, "regime": rg, "baseline": name, "selection_mode": mode,
                "metric": metric, "n_seeds": len(vals),
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })

    # ------------------------------------------ checkpoint-selection agreement
    # The baseline's weight and S_lambda's own lambda are not claimed to
    # correspond point-for-point (the ledger only claims curve-shape
    # agreement, not a shared numeric scale); what IS checkable is whether
    # the two parameterizations select the SAME checkpoint at their matching
    # endpoints -- weight=0 (pure mean-Brier) vs lambda=zero (pure base
    # rule, mean/sum-form differ only by a constant factor so argmin is
    # unaffected), and weight=1 (pure error rate) vs lambda=limit (the
    # lambda->infinity approximation, also pure error-rate-driven selection).
    def selected_epoch_map(rows, base, lambda_name):
        out = {}
        for r in rows:
            if r["base"] != base or r["lambda_name"] != lambda_name:
                continue
            key = (r["dataset"], r["regime"], int(r["seed"]), r["selection_mode"])
            out[key] = int(r["selected_epoch"])
        return out

    def agreement(map_a, map_b):
        keys = sorted(set(map_a) & set(map_b))
        matches = [map_a[k] == map_b[k] for k in keys]
        return {
            "n_pairs": len(keys),
            "n_agree": int(sum(matches)),
            "agreement_rate": float(np.mean(matches)) if keys else float("nan"),
            "disagreements": [
                {"cell": f"{k[0]}/{k[1]}/seed{k[2]}/{k[3]}",
                 "baseline_epoch": map_a[k], "family_epoch": map_b[k]}
                for k in keys if map_a[k] != map_b[k]
            ],
        }

    baseline_zero = selected_epoch_map(f1_rows, "baseline", "weighted_mix_0.0")
    baseline_one = selected_epoch_map(f1_rows, "baseline", "weighted_mix_1.0")
    checkpoint_agreement = {}
    for base in ["brier", "log"]:
        family_limit = selected_epoch_map(f1_rows, base, "limit")
        checkpoint_agreement[base] = {
            "weight1_vs_lambda_limit": agreement(baseline_one, family_limit),
        }
    # weight=0 in the baseline is hardcoded to mean-Brier (scoring.py
    # weighted_mixture_score), so comparing it against the LOG family's own
    # lambda=zero would silently compare two different base rules (Brier
    # argmin vs log-loss argmin), not the same rule at matching endpoints --
    # confirmed by an initial run where that comparison agreed only 56% of
    # the time, far below the >90% every genuinely-matching endpoint gives,
    # which is the signature of comparing unrelated rules rather than a
    # real disagreement. Only reported for the Brier base, where weight=0
    # and lambda=zero are actually the same rule.
    checkpoint_agreement["brier"]["weight0_vs_lambda_zero"] = agreement(
        baseline_zero, selected_epoch_map(f1_rows, "brier", "zero"))

    # ---------------------------------------------------- primary comparison:
    # star_plus vs published macro-F1, paired by (dataset, regime, seed, mode),
    # run once per base family (brier, log), Holm-corrected across the two.
    def paired_values(base, lam_a, lam_b, rows=f1_rows):
        a = {}
        b = {}
        for r in rows:
            if r["base"] != base:
                continue
            key = (r["dataset"], r["regime"], int(r["seed"]), r["selection_mode"])
            if r["lambda_name"] == lam_a:
                a[key] = float(r["value"])
            elif r["lambda_name"] == lam_b:
                b[key] = float(r["value"])
        keys = sorted(set(a) & set(b))
        return keys, [a[k] for k in keys], [b[k] for k in keys]

    primary = {}
    raw_pvals = []
    base_order = []
    for base in ["brier", "log"]:
        keys, star_plus_vals, published_vals = paired_values(base, "star_plus", "published")
        diffs = np.array(star_plus_vals) - np.array(published_vals)
        if np.allclose(diffs, 0.0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(star_plus_vals, published_vals, zero_method="wilcox",
                                alternative="two-sided")
            stat, p = float(stat), float(p)
        primary[base] = {
            "n_pairs": len(keys),
            "mean_star_plus": float(np.mean(star_plus_vals)),
            "std_star_plus": float(np.std(star_plus_vals, ddof=1)),
            "mean_published": float(np.mean(published_vals)),
            "std_published": float(np.std(published_vals, ddof=1)),
            "mean_diff_star_plus_minus_published": float(np.mean(diffs)),
            "std_diff_star_plus_minus_published": float(np.std(diffs, ddof=1)),
            "wilcoxon_statistic": stat, "p_raw": p,
        }
        raw_pvals.append(p)
        base_order.append(base)

    adj = holm(raw_pvals)
    for base, p_adj in zip(base_order, adj):
        primary[base]["p_holm"] = p_adj
        primary[base]["significant_at_alpha"] = bool(p_adj < ALPHA)

    # ---------------------------------------------------- primary comparison,
    # cell-level: the n=90 test above pairs by (dataset, regime, seed, mode),
    # but seed and selection_mode are both nested within just 9 (dataset,
    # regime) cells -- 5 seeds are independent training runs, but the two
    # selection modes are two ways of reading OFF THE SAME trained run, not
    # independent replicates, so treating all 90 as independent pairs
    # overstates the effective sample size and can make the test
    # anti-conservative. This collapses each cell's 10 (seed, mode) pairs to
    # one cell-level mean difference first, so the inferential unit is the
    # cell (n=9), not the individual pair; reported as a second analysis
    # alongside the original rather than a replacement for it.
    def cell_level_comparison(base, lam_a, lam_b, rows=f1_rows, exclude_datasets=()):
        keys, a_vals, b_vals = paired_values(base, lam_a, lam_b, rows)
        by_cell = defaultdict(list)
        for (d, rg, seed, mode), av, bv in zip(keys, a_vals, b_vals):
            if d in exclude_datasets:
                continue
            by_cell[(d, rg)].append(av - bv)
        cells = sorted(by_cell)
        cell_means = [float(np.mean(by_cell[c])) for c in cells]
        arr = np.array(cell_means)
        n_pos = int(np.sum(arr > 0))
        n_neg = int(np.sum(arr < 0))
        n_zero = int(np.sum(arr == 0))
        if np.allclose(arr, 0.0) or len(arr) < 2:
            w_stat, w_p = float("nan"), 1.0
        else:
            w_stat, w_p = wilcoxon(arr, alternative="two-sided",
                                    zero_method="wilcox")
            w_stat, w_p = float(w_stat), float(w_p)
        # exact two-sided sign test over non-zero cells, as a second, even
        # more assumption-light check at this small a sample size
        n_nonzero = n_pos + n_neg
        if n_nonzero == 0:
            sign_p = 1.0
        else:
            from scipy.stats import binomtest
            sign_p = float(binomtest(min(n_pos, n_neg), n_nonzero, 0.5,
                                      alternative="two-sided").pvalue)

        # Effect-size reporting (the primary evidence at this unit): mean
        # cell-level difference, its standard error, a t-based 95% CI (df =
        # n_cells-1), and a percentile bootstrap 95% CI (10000 resamples,
        # fixed seed) as an assumption-light cross-check, since n=9 makes
        # the t-interval's normality assumption worth corroborating rather
        # than trusting alone. Median difference given alongside the mean
        # since five (Brier) or three (log) of the nine cells are exactly
        # zero, which pulls the two apart.
        n = len(arr)
        mean_diff = float(np.mean(arr))
        median_diff = float(np.median(arr))
        std_diff = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
        if n > 1 and se_diff > 0:
            from scipy.stats import t as t_dist
            tcrit = float(t_dist.ppf(0.975, df=n - 1))
            ci_t = (mean_diff - tcrit * se_diff, mean_diff + tcrit * se_diff)
        else:
            ci_t = (mean_diff, mean_diff)
        rng = np.random.default_rng(20260906)
        boot_means = rng.choice(arr, size=(10000, n), replace=True).mean(axis=1)
        ci_boot = (float(np.percentile(boot_means, 2.5)),
                   float(np.percentile(boot_means, 97.5)))

        return {
            "n_cells": len(cells),
            "cells": [f"{d}/{rg}" for d, rg in cells],
            "cell_mean_diffs_star_plus_minus_published": cell_means,
            "n_cells_favoring_star_plus": n_pos,
            "n_cells_favoring_published": n_neg,
            "n_cells_tied": n_zero,
            "mean_diff": mean_diff, "median_diff": median_diff,
            "std_diff": std_diff, "se_diff": se_diff,
            "ci95_t": [ci_t[0], ci_t[1]],
            "ci95_bootstrap": [ci_boot[0], ci_boot[1]],
            "wilcoxon_statistic": w_stat, "wilcoxon_p_raw": w_p,
            "sign_test_p_raw": sign_p,
        }

    primary_cell_level = {}
    raw_pvals_cell = []
    for base in base_order:
        primary_cell_level[base] = cell_level_comparison(base, "star_plus", "published")
        raw_pvals_cell.append(primary_cell_level[base]["wilcoxon_p_raw"])
    adj_cell = holm(raw_pvals_cell)
    for base, p_adj in zip(base_order, adj_cell):
        primary_cell_level[base]["wilcoxon_p_holm"] = p_adj
        primary_cell_level[base]["significant_at_alpha"] = bool(p_adj < ALPHA)

    # Confirmatory (pre-specified, n=8) vs full (n=9, includes the
    # post-amendment emnist_balanced cell) cell-level analysis. The original
    # two-dataset design frozen before any result was read covered only
    # mnist and cifar10 (8 cells); emnist_balanced was added afterward by
    # Amendment 1, so pooling it into the "primary" n=9 statistic conflates
    # confirmatory and exploratory evidence. Reported separately here at a
    # referee's request; the n=9 version above is kept for continuity with
    # earlier drafts but the n=8 version is the one we now call confirmatory.
    primary_cell_level_confirmatory = {}
    raw_pvals_cell_confirmatory = []
    for base in base_order:
        primary_cell_level_confirmatory[base] = cell_level_comparison(
            base, "star_plus", "published", exclude_datasets=("emnist_balanced",))
        raw_pvals_cell_confirmatory.append(
            primary_cell_level_confirmatory[base]["wilcoxon_p_raw"])
    adj_cell_confirmatory = holm(raw_pvals_cell_confirmatory)
    for base, p_adj in zip(base_order, adj_cell_confirmatory):
        primary_cell_level_confirmatory[base]["wilcoxon_p_holm"] = p_adj
        primary_cell_level_confirmatory[base]["significant_at_alpha"] = bool(p_adj < ALPHA)

    report = {
        "provenance": provenance,
        "informativeness": informativeness,
        "diagnostics_by_regime": diagnostics,
        "frontier": frontier,
        "stability": stability,
        "baselines": baselines,
        "primary_comparison": primary,
        "primary_comparison_cell_level": primary_cell_level,
        "primary_comparison_cell_level_confirmatory_n8": primary_cell_level_confirmatory,
        "checkpoint_agreement": checkpoint_agreement,
        "alpha": ALPHA,
    }

    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"provenance: {provenance}")
    print(f"informative regimes: {sum(1 for v in informativeness.values() if v['informative'])}/{len(informativeness)}")
    for k, v in sorted(informativeness.items()):
        print(f"  {k}: mean achieved macro-F1 = {v['mean_achieved_macro_f1']:.4f}  informative={v['informative']}")
    print("primary comparison (star_plus vs published, macro-F1):")
    for base, res in primary.items():
        print(f"  {base}: n={res['n_pairs']} mean_diff={res['mean_diff_star_plus_minus_published']:.5f} "
              f"p_raw={res['p_raw']:.4g} p_holm={res['p_holm']:.4g} significant={res['significant_at_alpha']}")
    print("primary comparison, cell-level (9 cells as the inferential unit):")
    for base, res in primary_cell_level.items():
        print(f"  {base}: n_cells={res['n_cells']} favor_star_plus={res['n_cells_favoring_star_plus']} "
              f"favor_published={res['n_cells_favoring_published']} tied={res['n_cells_tied']} "
              f"wilcoxon_p_raw={res['wilcoxon_p_raw']:.4g} wilcoxon_p_holm={res['wilcoxon_p_holm']:.4g} "
              f"sign_p_raw={res['sign_test_p_raw']:.4g} significant={res['significant_at_alpha']}")


if __name__ == "__main__":
    main()
