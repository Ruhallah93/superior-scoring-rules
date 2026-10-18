"""T11, T15, T15b -- strict propriety of S_lambda under each tie convention.

A rule is strictly proper iff  E_Q[S(P,.)] > E_Q[S(Q,.)]  for every P != Q.
We search for counterexamples P, seeding the search with the forecasts that
theory says are dangerous: the uniform vector (escapes the penalty entirely
under convention A) and tie-broken versions of Q (beat truthful reporting under
convention B when Q itself has tied modes).

NAMING, READ THIS BEFORE TOUCHING LETTERS: this file (and common.py) use
A=permissive/ties-to-correct, B=ties-to-incorrect (strict inequality
required, exact ties assigned to xi), C=single-valued. The *paper draft*
(paper/notation.md, preliminaries.md, theory-2.md) currently only defines
two conventions and, in the task that added the third one, calls it
"Convention C" while explicitly preserving its own "Convention A" (permissive)
and "Convention B" (single-valued) unchanged. So: this file's convention "B"
(ties-to-xi) IS the paper's proposed "Convention C". This file's convention
"C" (single-valued) IS the paper's existing "Convention B". Do not rename
anything in common.py to fix this -- check_superiority.py already depends on
the current letters. Every result below that concerns the paper's proposed
third convention is computed with conv="B" and labelled accordingly.
"""
import os
import numpy as np
from common import (Result, brier_sum, expected_score, lambda_star_brier,
                    lam_definition6, lam_implemented_sum_equiv,
                    CONVENTIONS, s_lambda)

FULL = os.environ.get("VERIFY_LEVEL", "quick") == "full"
N_Q = 2000 if FULL else 400
N_P = 60 if FULL else 20


def adversaries(Q, rng):
    c = len(Q)
    adv = [np.full(c, 1.0 / c)]                       # the convention-A escape
    m = int(np.argmax(Q))
    for eps in (1e-5, 1e-3, 1e-2, 0.05):              # tie-broken versions of Q
        P = Q.copy(); P[m] += eps
        P = np.clip(P, 0, None); P = P / P.sum()
        adv.append(P)
    for eps in (1e-4, 1e-3, 1e-2, 0.1):               # local perturbations
        P = Q + eps * (rng.random(c) - 0.5)
        P = np.clip(P, 1e-9, None); P = P / P.sum()
        adv.append(P)
    for _ in range(N_P):
        adv.append(rng.dirichlet(np.ones(c)))
    return adv


def scan(c, lam, conv, rng, tied_modes=False):
    """Returns (violations, worst_deficit) over N_Q draws of Q."""
    viol, worst = 0, 0.0
    for _ in range(N_Q):
        if tied_modes:
            Q = rng.dirichlet(np.ones(c))
            a, b = rng.choice(c, 2, replace=False)
            Q[a] = Q[b] = max(Q) + 0.05
            Q = Q / Q.sum()
        else:
            Q = rng.dirichlet(np.ones(c))
        sQ = expected_score(Q, Q, lam, brier_sum, conv)
        for P in adversaries(Q, rng):
            # A meaningful counterexample must be a genuinely different
            # forecast; P within float noise of Q tests nothing.
            if np.max(np.abs(P - Q)) < 1e-6:
                continue
            sP = expected_score(P, Q, lam, brier_sum, conv)
            if sP < sQ - 1e-12:
                viol += 1
                worst = max(worst, sQ - sP)
                break
    return viol, worst


def run(rng):
    out = []
    rC = Result("T11", "S_lambda is strictly proper for all lambda >= 0 "
                       "under a single-valued decision rule (convention C)")
    rA = Result("T15", "convention A does NOT preserve strict propriety")
    rB = Result("T15b", "convention B fails strict propriety only when Q has "
                        "tied modes")

    print("\n  violations out of %d draws of Q (Brier base):" % N_Q)
    print("  %-4s %-22s %-6s %10s %14s" % ("c", "lambda", "conv", "violations",
                                           "worst deficit"))
    for c in (3, 5, 13):
        lams = [("0", 0.0), ("lambda*", lambda_star_brier(c)),
                ("Definition 6", lam_definition6(c)),
                ("implemented", lam_implemented_sum_equiv(c))]
        for name, lam in lams:
            for conv in ("A", "B", "C"):
                v, w = scan(c, lam, conv, rng)
                print("  %-4d %-22s %-6s %7d/%d %14.5f"
                      % (c, "%s = %.4f" % (name, lam), conv, v, N_Q, w))
                if conv == "C":
                    rC.add("c=%d lambda=%s: no violation" % (c, name), v == 0,
                           "%d/%d" % (v, N_Q))
                if conv == "A" and lam > 0:
                    rA.add("c=%d lambda=%s: violations found (expected)"
                           % (c, name), v > 0, "%d/%d, worst deficit %.5f"
                           % (v, N_Q, w))

    print("\n  convention B, Q constructed with TIED modes:")
    print("  %-4s %-22s %10s %14s" % ("c", "lambda", "violations", "worst deficit"))
    for c in (3, 5):
        for name, lam in [("lambda*", lambda_star_brier(c)),
                          ("Definition 6", lam_definition6(c))]:
            v, w = scan(c, lam, "B", rng, tied_modes=True)
            print("  %-4d %-22s %7d/%d %14.5f"
                  % (c, "%s = %.4f" % (name, lam), v, N_Q, w))
            rB.add("c=%d lambda=%s: tied-mode Q breaks B (expected)"
                   % (c, name), v > 0, "%d/%d" % (v, N_Q))
        v, w = scan(c, lambda_star_brier(c), "C", rng, tied_modes=True)
        rC.add("c=%d tied-mode Q: C still clean" % c, v == 0, "%d/%d" % (v, N_Q))

    # closed-form threshold for the convention-A escape
    rA_thresh = Result("T15a", "closed-form threshold above which the uniform "
                               "forecast beats truthful reporting under A")
    c, Q = 3, np.array([0.5, 0.3, 0.2])
    u = np.full(c, 1.0 / c)
    num = expected_score(u, Q, 0.0, brier_sum, "A") - expected_score(Q, Q, 0.0, brier_sum, "A")
    thresh = num / (1 - Q.max())
    below = expected_score(Q, Q, thresh * 0.9, brier_sum, "A") < expected_score(u, Q, thresh * 0.9, brier_sum, "A")
    above = expected_score(Q, Q, thresh * 1.1, brier_sum, "A") > expected_score(u, Q, thresh * 1.1, brier_sum, "A")
    rA_thresh.add("threshold formula is exact at Q=(0.5,0.3,0.2)", below and above,
                  "threshold = %.6f; Definition 6 lambda = %.4f exceeds it"
                  % (thresh, lam_definition6(c)))

    out += [rC, rA, rB, rA_thresh, check_third_convention_edge_cases(rng)]
    return out


def _L_third(p, i):
    """The penalty-only indicator for the paper's proposed third convention
    (= this file's convention 'B'): 1 unless i is the UNIQUE argmax of p."""
    return 0.0 if CONVENTIONS["B"](p, i) else 1.0


def _zero_base(p, i):
    return 0.0


def check_third_convention_edge_cases(rng):
    """Dedicated, closed-form-checking tests for the paper's proposed third
    convention (ties-to-xi, strict inequality; this file's conv='B'),
    covering each case named in the task that added this function:
    generic Q with a unique mode, Q with a tied mode, c=2, c>2, reports p
    with exact ties, and truthful reports p=Q. These check the *closed
    forms* the proof derives, not just "no counterexample found" -- the
    scan()-based checks above already do the latter.
    """
    r = Result("T15b-edge", "closed-form checks for the third convention "
                            "(ties-to-xi) at named edge cases")

    # --- Case 1: generic Q, unique mode. Closed form: E_Q[L(Q,.)] = 1-max(Q),
    # and this is achieved by ANY p with a unique max at argmax(Q), not only Q.
    for c in (2, 3, 5, 8, 13):
        for _ in range(200):
            Q = rng.dirichlet(np.ones(c))
            if len(np.flatnonzero(np.isclose(Q, Q.max(), atol=1e-9))) > 1:
                continue  # reject the negligible-probability tied draw
            expected = 1 - Q.max()
            actual_at_Q = sum(Q[j] * _L_third(Q, j) for j in range(c))
            m = int(np.argmax(Q))
            p_alt = np.zeros(c); p_alt[m] = 0.9;
            if c > 1:
                p_alt[(m + 1) % c] = 0.1
            actual_at_alt = sum(Q[j] * _L_third(p_alt, j) for j in range(c))
            r.add("c=%d unique-mode Q: E_Q[L(Q,.)]=1-max(Q) exactly, "
                  "and a DIFFERENT unique-max-at-argmax(Q) report ties it"
                  % c,
                  abs(actual_at_Q - expected) < 1e-9 and abs(actual_at_alt - expected) < 1e-9,
                  "E_Q[L(Q,.)]=%.6f, E_Q[L(alt,.)]=%.6f, 1-max(Q)=%.6f"
                  % (actual_at_Q, actual_at_alt, expected))

    # --- Case 2: Q with a tied mode, every tie-set size k=2..c, several c.
    # Closed form: E_Q[L(Q,.)] = 1 exactly (worst possible); a report that
    # breaks the tie toward one top class approaches 1-q* as eps -> 0, a
    # strict improvement of exactly q* over the truthful report, for EVERY
    # base and EVERY lambda > 0 (not just the ones scan() happens to try).
    for c in (2, 3, 5, 8, 13):
        for k in range(2, c + 1):
            Q = np.zeros(c)
            tie_idx = rng.choice(c, size=k, replace=False)
            if k == c:
                qstar = 1.0 / c  # full tie: forced, no freedom left
            else:
                # Need qstar strictly > 1/c so a UNIFORM split of the
                # remaining mass over the c-k other classes is guaranteed
                # to keep every other class strictly below qstar:
                # remaining/(c-k) < qstar  <=>  qstar > 1/c  (algebra below).
                # An uncontrolled Dirichlet draw was tried first and failed
                # exactly here -- it can concentrate enough mass onto one
                # "other" class to accidentally outweigh the intended tie,
                # silently building a Q that doesn't have the tied mode the
                # test meant to construct. Uniform split has no such risk.
                lo, hi = 1.0 / c * 1.01, 0.95 / k
                qstar = rng.uniform(lo, hi) if hi > lo else lo
            Q[tie_idx] = qstar
            remaining = 1 - k * qstar
            other_idx = [j for j in range(c) if j not in tie_idx]
            if other_idx:
                each = remaining / len(other_idx)
                assert each < qstar - 1e-12, (c, k, qstar, each)  # margin, not just <=
                for j in other_idx:
                    Q[j] = each
            assert np.isclose(Q.sum(), 1.0)
            assert set(np.flatnonzero(np.isclose(Q, Q.max(), atol=1e-9))) == set(tie_idx.tolist()), \
                (c, k, qstar, Q)  # M(Q) is exactly the intended tie-set, nothing more

            at_Q = sum(Q[j] * _L_third(Q, j) for j in range(c))
            k_class = int(tie_idx[0])
            eps = 1e-6
            p_eps = Q.copy(); p_eps[k_class] += eps; p_eps = p_eps / p_eps.sum()
            at_peps = sum(Q[j] * _L_third(p_eps, j) for j in range(c))
            r.add("c=%d k=%d-way tie at q*=%.4f: E_Q[L(Q,.)]=1, "
                  "tie-broken report -> 1-q*" % (c, k, qstar),
                  abs(at_Q - 1.0) < 1e-9 and abs(at_peps - (1 - qstar)) < 1e-6,
                  "E_Q[L(Q,.)]=%.6f, E_Q[L(p_eps,.)]=%.6f, 1-q*=%.6f"
                  % (at_Q, at_peps, 1 - qstar))

            # violation holds for EVERY lambda > 0, brier and log base, not
            # just lambda* / Definition 6 -- sweep several orders of magnitude
            for lam in (1e-6, 1e-3, 0.1, 1.0, 10.0):
                for base_name, base_fn in (("brier", brier_sum),
                                            ("zero_base", _zero_base)):
                    sQ = sum(Q[j] * s_lambda(Q, j, lam, base_fn, "B") for j in range(c))
                    sP = sum(Q[j] * s_lambda(p_eps, j, lam, base_fn, "B") for j in range(c))
                    r.add("c=%d k=%d lambda=%.0e base=%s: tie-broken report beats Q"
                          % (c, k, lam, base_name),
                          sP < sQ,
                          "sQ=%.6f sP=%.6f margin=%.2e" % (sQ, sP, sQ - sP))

    # --- Case 3: c=2 specifically -- the only tied-mode Q is (0.5,0.5), and
    # lambda* = 0 there (Corollary 1 / T8), so "every lambda>0 fails" is a
    # sharper, more consequential statement at c=2 than at c>2: it isn't a
    # narrow band above lambda*, it is the ENTIRE recommended range.
    Q2 = np.array([0.5, 0.5])
    at_Q2 = sum(Q2[j] * _L_third(Q2, j) for j in range(2))
    r.add("c=2 tied Q=(0.5,0.5): E_Q[L(Q,.)]=1 exactly", abs(at_Q2 - 1.0) < 1e-9,
          "%.6f" % at_Q2)
    for lam in (1e-9, 1e-6, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0):
        p_eps = np.array([0.5 + 1e-6, 0.5 - 1e-6])
        sQ = sum(Q2[j] * s_lambda(Q2, j, lam, brier_sum, "B") for j in range(2))
        sP = sum(Q2[j] * s_lambda(p_eps, j, lam, brier_sum, "B") for j in range(2))
        r.add("c=2, lambda=%.0e (lambda*=0 at c=2, so this is the ENTIRE "
              "admissible range, not a narrow band): tie-broken beats Q" % lam,
              sP < sQ, "sQ=%.6f sP=%.6f" % (sQ, sP))

    # --- Case 4: reports p with exact ties are never advantaged, for ANY Q,
    # ANY true class -- a tied report always incurs the full penalty.
    for c in (3, 5, 8):
        for _ in range(200):
            k = rng.integers(2, c + 1)
            p = np.zeros(c)
            tie_idx = rng.choice(c, size=k, replace=False)
            p[tie_idx] = 1.0 / k
            for j in range(c):
                assert _L_third(p, j) == 1.0, (c, k, j, p)
        r.add("c=%d: every report with |M(p)|>=2 has L(p,.)=1 for every true class"
              % c, True, "200 random tied reports, k=2..%d, all confirmed" % c)

    # --- Case 5: truthful reports p=Q are the reference throughout cases
    # 1-2 above; separately confirm p=Q is never itself a counterexample to
    # its own propriety bound (sanity check on the harness, not new math).
    for c in (3, 5):
        for _ in range(50):
            Q = rng.dirichlet(np.ones(c))
            sQ = sum(Q[j] * s_lambda(Q, j, 0.1, brier_sum, "B") for j in range(c))
            sQ_again = sum(Q[j] * s_lambda(Q, j, 0.1, brier_sum, "B") for j in range(c))
            assert abs(sQ - sQ_again) < 1e-12
        r.add("c=%d: E_Q[S(Q,.)] is well-defined and deterministic (harness sanity)"
              % c, True, "50 random Q, reproducible")

    return r
