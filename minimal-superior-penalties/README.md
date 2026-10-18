# Minimal Superior Penalties

Code and results for a follow-up study that revisits the penalty introduced
above. It identifies the exact threshold a penalty must clear for
*superiority* to hold, corrects an error in the original strict-propriety
proof, and reports a pre-specified empirical case study on what the
published penalty costs relative to the minimal one.

Manuscript: *Minimal Superior Penalties: Characterising and Calibrating
Decision-Aware Proper Scoring Rules* (Rouhollah Ahmadian). Submitted
independently of the original paper and its other co-authors.

## What this adds

1. **The infimal penalty.** Superiority is governed by the gap between two
   closed-form quantities -- the supremum of the base score on correct
   predictions and the infimum on incorrect predictions. This gap, the
   *infimal penalty*, is the exact threshold a penalty must clear. The
   published penalty (above) exceeds this threshold by a provable factor and
   is one point within a strictly proper one-parameter family, not a
   uniquely determined value.

2. **A corrected propriety proof.** The strict-propriety argument for the
   published penalty had an error under certain tie-breaking conventions.
   This study corrects it and generalises the result to the whole penalty
   family under an explicit, single-valued tie-breaking convention.

3. **A pre-specified empirical case study**, reported with its limitations
   stated plainly rather than minimized: the observed effect at ten classes
   is not statistically distinguishable from noise once the experimental
   design's nesting structure is accounted for, a fifth regime at
   forty-seven classes does not reproduce it, and an intended
   no-superiority control baseline turned out, on inspection, to be
   algebraically inseparable from the family under test. The design
   limitation is reported as a central finding, not a footnote.

## Layout

```
minimal-superior-penalties/
├── experiments/
│   ├── config.yaml              # pre-registered experimental design
│   ├── requirements.txt
│   └── pipeline/                # data loading, models, training, scoring, selection
├── verify/                      # independent checks: superiority, propriety, published-value audit
├── figures/
│   └── make_figures.py          # regenerates every paper figure from results/ and verify/
└── results/                     # raw per-run metrics (ece, f1, nll, margins, ...)
```

`experiments/config.yaml` is the frozen pre-registration: the lambda grid,
datasets, regimes, outcome measures, baselines, and decision rule were all
fixed before any result was read. Later additions are recorded in the file
itself as dated `AMENDMENT` notes rather than silently edited in.

## Reproducing

```bash
pip install -r experiments/requirements.txt
python experiments/pipeline/run_experiments.py   # trains/evaluates per config.yaml, writes results/
python verify/run_all.py                         # independent checks (superiority, propriety, published value)
python figures/make_figures.py                   # regenerates figures/fig{1..5}_*.{pdf,png}
```

## Code and data availability

Everything behind every number reported in the manuscript -- experiment
configuration, training/scoring pipeline, independent verification checks,
figure generation, and raw results -- is in this folder.
