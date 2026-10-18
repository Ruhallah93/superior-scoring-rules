"""E1-E6 -- claims about the published paper and the released implementation.

The PBS transcription below is a faithful numpy port of the released
TensorFlow function, including its ceil-based tie test.
"""
import numpy as np
from common import (Result, close, brier_sum, brier_mean, lam_definition6,
                    lam_implemented_sum_equiv, lambda_star_brier, sup_psi_brier)


def pbs_released(y, q):
    """Faithful port of superior_scoring_rules.pbs (mean-form base)."""
    y = y.astype(np.float64); c = y.shape[1]
    ST = q - np.sum(np.where(y == 1, q, y), axis=1)[:, None]
    ST = np.where(ST < 0, 0.0, ST)
    payoff = np.sum(np.ceil(ST), axis=1)
    payoff = np.where(payoff > 0, (c - 1) / c, payoff)
    return np.mean(np.square(y - q), axis=1) + payoff


def pbs_definition6(y, q):
    """Source Eq. (42): SUM-form base, same penalty."""
    c = y.shape[1]
    wrong = np.argmax(q, 1) != np.argmax(y, 1)
    return np.sum(np.square(y - q), 1) + np.where(wrong, (c - 1) / c, 0.0)


def run(rng):
    out = []

    # ------------------------------------------------------------ E1 / E1a
    r1 = Result("E1", "source Eq. (46) S_BS(Q,Q) = S_PBS(Q,Q) does not hold")
    r1a = Result("E1a", "corrected identity: gap equals lambda(1 - max_j Q_j)")
    cases = [(3, np.array([0.5, 0.3, 0.2])), (3, np.array([0.4, 0.35, 0.25])),
             (4, np.array([0.7, 0.1, 0.1, 0.1])), (5, np.array([0.4, 0.2, 0.2, 0.1, 0.1]))]
    for c, Q in cases:
        lam = lam_definition6(c); m = int(np.argmax(Q))
        e_bs = sum(Q[i] * brier_sum(Q, i) for i in range(c))
        e_pbs = sum(Q[i] * (brier_sum(Q, i) + (lam if i != m else 0.0)) for i in range(c))
        r1.add("c=%d Q=%s: gap is nonzero" % (c, np.round(Q, 2)),
               abs(e_pbs - e_bs) > 1e-9, "gap %.6f" % (e_pbs - e_bs))
        r1a.add("c=%d: gap = lambda(1-maxQ)" % c,
                close(e_pbs - e_bs, lam * (1 - Q.max())),
                "%.6f vs %.6f" % (e_pbs - e_bs, lam * (1 - Q.max())))
    # degenerate forecast is the only equality case
    Qd = np.array([1.0, 0.0, 0.0])
    e_bs = sum(Qd[i] * brier_sum(Qd, i) for i in range(3))
    e_pbs = e_bs + lam_definition6(3) * (1 - Qd.max())
    r1a.add("equality only when max_j Q_j = 1", close(e_pbs, e_bs),
            "degenerate Q gives gap 0")

    # -------------------------------------------------------- E2 / E3 / E3a
    r3 = Result("E3", "released code matches Algorithm 2; sum-equivalent "
                      "penalty is c-1")
    r3a = Result("E3a", "ratios of published and implemented penalties to lambda*")
    r3b = Result("E3b", "implemented penalty exceeds the whole range of its "
                        "own calibration term by a factor of c")
    print("\n  %-4s %10s %12s %10s %11s %11s %13s"
          % ("c", "Def6 lam", "impl lam", "lambda*", "Def6/lam*", "impl/lam*",
             "impl/range_psi"))
    for c in (3, 5, 10, 12, 13, 20):
        d6, im, ls = lam_definition6(c), lam_implemented_sum_equiv(c), lambda_star_brier(c)
        print("  %-4d %10.4f %12.4f %10.4f %11.2f %11.2f %13.2f"
              % (c, d6, im, ls, d6 / ls, im / ls, im / sup_psi_brier(c)))
        # confirm the sum-equivalence by execution
        y = np.eye(c)[[0]]
        q = np.zeros((1, c)); q[0, 1] = 0.6; q[0, 0] = 0.4
        released = pbs_released(y, q)[0] * c            # rescale to sum form
        expected = brier_sum(q[0], 0) + (c - 1)
        r3.add("c=%d: released*c equals sum-Brier + (c-1)" % c,
               close(released, expected), "%.6f vs %.6f" % (released, expected))
        r3a.add("c=%d ratios finite and > 1" % c, d6 / ls > 1 and im / ls > 1,
                "Def6/lam*=%.2f impl/lam*=%.2f" % (d6 / ls, im / ls))
        r3b.add("c=%d: impl / sup_psi = c" % c,
                close(im / sup_psi_brier(c), c), "%.4f" % (im / sup_psi_brier(c)))

    # E2: Definition 6 and Algorithm 2 disagree
    r2 = Result("E2", "Definition 6 (sum base) and Algorithm 2 (mean base) "
                      "use the same penalty and therefore differ")
    for c in (3, 13):
        y = np.eye(c)[[0]]
        q = np.zeros((1, c)); q[0, 1] = 0.6; q[0, 0] = 0.4
        r2.add("c=%d: released != Definition 6" % c,
               not close(pbs_released(y, q)[0], pbs_definition6(y, q)[0]),
               "released %.4f vs Def6 %.4f"
               % (pbs_released(y, q)[0], pbs_definition6(y, q)[0]))

    # ------------------------------------------------------------ E5 / E5a
    r5 = Result("E5", "source Table 1 log-loss values are base 10")
    r5a = Result("E5a", "source Table 1 Brier values are sum form")
    A = np.array([0.33, 0.34, 0.33]); B = np.array([0.51, 0.49, 0.0])
    r5.add("A: published 0.4685 equals -log10(0.34)",
           close(-np.log10(0.34), 0.4685, 1e-4),
           "-log10=%.4f  -ln=%.4f" % (-np.log10(0.34), -np.log(0.34)))
    r5.add("B: published 0.3098 equals -log10(0.49)",
           close(-np.log10(0.49), 0.3098, 1e-4),
           "-log10=%.4f  -ln=%.4f" % (-np.log10(0.49), -np.log(0.49)))
    r5a.add("A: published 0.6534 equals sum-form Brier",
            close(brier_sum(A, 1), 0.6534, 1e-4),
            "sum=%.4f mean=%.4f" % (brier_sum(A, 1), brier_mean(A, 1)))
    r5a.add("B: published 0.5202 equals sum-form Brier",
            close(brier_sum(B, 1), 0.5202, 1e-4),
            "sum=%.4f mean=%.4f" % (brier_sum(B, 1), brier_mean(B, 1)))
    r5a.add("B lies above the inf_xi bound of 1/2", brier_sum(B, 1) > 0.5,
            "%.4f > 0.5 -- B is the near-minimiser of T2" % brier_sum(B, 1))

    # ----------------------------------------------------------------- E6
    r6 = Result("E6", "released code resolves exact argmax ties in favour of "
                      "the correct set")
    for c in (3, 13):
        y = np.eye(c)[[1]]
        q = np.full((1, c), 1.0 / c)                    # uniform: a c-way tie
        released = pbs_released(y, q)[0]
        unpenalised = close(released, brier_mean(q[0], 1))
        r6.add("c=%d: uniform forecast is not penalised" % c, unpenalised,
               "released %.6f, mean-Brier alone %.6f"
               % (released, brier_mean(q[0], 1)))
    r6.add("mechanism: ceil(0) == 0", close(float(np.ceil(0.0)), 0.0),
           "the ceil-based test cannot fire on an exact tie")

    out += [r1, r1a, r2, r3, r3a, r3b, r5, r5a, r6]
    return out
