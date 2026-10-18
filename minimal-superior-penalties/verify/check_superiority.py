"""T13 / Theorem M3 -- for which lambda is S_lambda superior?

Superiority quantifies over ALL pairs (p1, w1) in psi and (p2, w2) in xi,
including pairs with different true classes.  So the test is:

    max over the union of psi   of S_lambda   <   min over the union of xi.

The extremisers are argmax ties, which random sampling will never hit, so the
candidate set explicitly contains the uniform vector, the 1/2-tie vectors, and
perturbations of both.
"""
import itertools
import os
import numpy as np
from common import (Result, brier_sum, lambda_star_brier, lam_definition6,
                    lam_implemented_sum_equiv, s_lambda, CONVENTIONS,
                    CONVENTION_NAMES)


FULL = os.environ.get("VERIFY_LEVEL", "quick") == "full"
N_RANDOM = 3000 if FULL else 600


def candidates(c, rng, n_random=None):
    n_random = N_RANDOM if n_random is None else n_random
    cands = [np.full(c, 1.0 / c)]                       # uniform: sup_psi
    for a in range(min(c, 3)):
        for b in range(c):
            if a != b:
                v = np.zeros(c); v[a] = v[b] = 0.5
                cands.append(v)                         # 1/2-tie: inf_xi
    for eps in (1e-9, 1e-6, 1e-3):
        for a, b in ((0, 1), (1, 0)):
            v = np.zeros(c); v[a] = 0.5 + eps; v[b] = 0.5 - eps
            cands.append(v)
        w = np.full(c, 1.0 / c - eps / (c - 1)); w[0] = 1.0 / c + eps
        cands.append(w)
        w = np.full(c, 1.0 / c + eps / (c - 1)); w[0] = 1.0 / c - eps
        cands.append(w)
    for _ in range(n_random):
        cands.append(rng.dirichlet(np.ones(c)))
    return cands


def margin(c, lam, conv, rng, base=brier_sum):
    """min over xi minus max over psi.  Superior iff strictly positive."""
    hi, lo = -np.inf, np.inf
    arg_hi = arg_lo = None
    for p in candidates(c, rng):
        for w in range(c):
            s = s_lambda(p, w, lam, base, conv)
            if CONVENTIONS[conv](p, w):
                if s > hi: hi, arg_hi = s, (p.copy(), w)
            else:
                if s < lo: lo, arg_lo = s, (p.copy(), w)
    return lo - hi, arg_hi, arg_lo


def make_delta(order):
    """delta(p) = the order-least element of argmax(p)."""
    rank = {cls: r for r, cls in enumerate(order)}

    def delta(p):
        m = p.max()
        return min((j for j in range(len(p)) if p[j] >= m - 1e-12),
                   key=lambda j: rank[j])
    return delta


def margin_order_induced(c, order, lam, base=brier_sum):
    """Margin at a given lambda under an order-induced tie-break.

    Uses only the extremal candidates -- the uniform vector and every
    1/2-tie -- because Corollary M6's construction lives entirely there.
    """
    delta = make_delta(order)
    cands = [np.full(c, 1.0 / c)]
    for a, b in itertools.permutations(range(c), 2):
        v = np.zeros(c); v[a] = v[b] = 0.5
        cands.append(v)
    hi, lo = -np.inf, np.inf
    for p in cands:
        for w in range(c):
            s = base(p, w) + (0.0 if delta(p) == w else lam)
            if delta(p) == w:
                hi = max(hi, s)
            else:
                lo = min(lo, s)
    return lo - hi


def run(rng):
    out = []
    r = Result("T13", "S_lambda is superior iff lambda exceeds lambda* "
                      "(boundary case depends on the tie convention)")
    rM = Result("M2", "attainment dichotomy: which extremum is reached")
    rP = Result("E-pos", "the published PBS penalty does induce superiority")

    table = []
    for c in [3, 5, 13]:
        ls = lambda_star_brier(c)
        for conv in ("A", "B", "C"):
            m_below, _, _ = margin(c, ls - 1e-3, conv, rng)
            m_at, hi_at, lo_at = margin(c, ls, conv, rng)
            m_above, _, _ = margin(c, ls + 1e-6, conv, rng)
            m_pub, _, _ = margin(c, lam_definition6(c), conv, rng)
            table.append((c, conv, m_below, m_at, m_above, m_pub))
            r.add("c=%d conv=%s: lambda < lambda* is NOT superior" % (c, conv),
                  m_below <= 0, "margin %.3e" % m_below)
            r.add("c=%d conv=%s: lambda > lambda* IS superior" % (c, conv),
                  m_above > 0, "margin %.3e" % m_above)
            rP.add("c=%d conv=%s: published lambda is superior" % (c, conv),
                   m_pub > 0, "margin %.3e" % m_pub)

    # The boundary case, reported per convention rather than asserted globally.
    print("\n  lambda = lambda* exactly -- margin (min_xi - max_psi):")
    print("  %-4s %-4s %14s   %s" % ("c", "conv", "margin", "superior at lambda*?"))
    boundary = {}
    for c in [3, 5, 13]:
        ls = lambda_star_brier(c)
        for conv in ("A", "B", "C"):
            m, hi, lo = margin(c, ls, conv, rng)
            boundary[(c, conv)] = m
            print("  %-4d %-4s %14.6e   %s" %
                  (c, conv, m, "yes" if m > 0 else "NO"))

    r.add("convention A admits lambda = lambda*",
          all(boundary[(c, "A")] > 0 for c in [3, 5, 13]),
          "margins " + ", ".join("%.2e" % boundary[(c, "A")] for c in [3, 5, 13]))
    r.add("convention B admits lambda = lambda*",
          all(boundary[(c, "B")] > 0 for c in [3, 5, 13]),
          "margins " + ", ".join("%.2e" % boundary[(c, "B")] for c in [3, 5, 13]))
    r.add("convention C does NOT admit lambda = lambda* (expected failure)",
          all(boundary[(c, "C")] <= 0 for c in [3, 5, 13]),
          "margins " + ", ".join("%.2e" % boundary[(c, "C")] for c in [3, 5, 13]))

    rM.add("A: sup_psi attained (uniform in psi), inf_xi not",
           CONVENTIONS["A"](np.full(3, 1 / 3), 0)
           and not CONVENTIONS["A"](np.array([.5, .5, 0.]), 0) is False,
           "uniform in psi_A: %s" % CONVENTIONS["A"](np.full(3, 1 / 3), 0))
    rM.add("B: uniform is NOT in psi_B",
           not CONVENTIONS["B"](np.full(3, 1 / 3), 0), "strict argmax required")
    rM.add("B: the 1/2-tie IS in xi_B for both its classes",
           not CONVENTIONS["B"](np.array([.5, .5, 0.]), 0)
           and not CONVENTIONS["B"](np.array([.5, .5, 0.]), 1), "ties -> xi")
    rM.add("C: uniform in psi_C for w=0 AND 1/2-tie in xi_C for w=1 "
           "-> both extrema attained",
           CONVENTIONS["C"](np.full(3, 1 / 3), 0)
           and not CONVENTIONS["C"](np.array([.5, .5, 0.]), 1),
           "this is why C needs lambda > lambda*")

    # ---------------------------------------------------------------- M6
    r6 = Result("M6", "no order-induced tie-break admits lambda = lambda*")
    print("\n  Corollary M6 -- margin at lambda* by tie-break order:")
    print("  %-4s %-16s %16s  %s" % ("c", "order", "margin", "superior?"))
    for c in (3, 5, 8, 13):
        orders = {
            "lowest-index": list(range(c)),
            "highest-index": list(range(c))[::-1],
            "middle-first": [c // 2] + [j for j in range(c) if j != c // 2],
            "random": [int(x) for x in rng.permutation(c)],
        }
        for name, order in orders.items():
            m = margin_order_induced(c, order, lambda_star_brier(c))
            print("  %-4d %-16s %16.3e  %s" % (c, name, m, "yes" if m > 0 else "NO"))
            r6.add("c=%d order=%s: lambda* NOT admissible (expected)" % (c, name),
                   m <= 0, "margin %.3e" % m)
            # and lambda strictly above lambda* does work
            m2 = margin_order_induced(c, order, lambda_star_brier(c) + 1e-6)
            r6.add("c=%d order=%s: lambda > lambda* IS superior" % (c, name),
                   m2 > 0, "margin %.3e" % m2)

    out += [r, rM, rP, r6]
    return out
