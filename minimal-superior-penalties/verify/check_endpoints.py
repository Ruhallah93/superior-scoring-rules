"""T1, T2, T2a, T3, T4, T6, T7 -- the four range endpoints and lambda*.

Method: constrained optimisation over the simplex (SLSQP, many restarts),
cross-checked against an independent dense grid on the reduced problem.
"""
import os
import numpy as np
from scipy.optimize import minimize
from common import (Result, close, brier_sum, sup_psi_brier, inf_xi_brier,
                    sup_psi_log, inf_xi_log, lambda_star_brier, lambda_star_log)

FULL = os.environ.get("VERIFY_LEVEL", "quick") == "full"
C_VALUES = ([3, 4, 5, 6, 8, 10, 12, 13, 20, 50, 100] if FULL
            else [3, 5, 13, 20])
N_RESTARTS = 40 if FULL else 12
GRID_N = 2_000_001 if FULL else 200_001
GRID_C = [3, 5, 13, 50] if FULL else [3, 13]


def _opt_brier(c, i, maximise, extra_cons, rng):
    sign = -1.0 if maximise else 1.0
    cons = [{"type": "eq", "fun": lambda p: p.sum() - 1.0}] + extra_cons
    best = None
    for _ in range(N_RESTARTS):
        x0 = rng.dirichlet(np.ones(c))
        r = minimize(lambda p: sign * brier_sum(p, i), x0, constraints=cons,
                     bounds=[(0, 1)] * c, method="SLSQP",
                     options={"maxiter": 800, "ftol": 1e-14})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return sign * best.fun, best.x


def sup_psi_numeric(c, rng, i=0):
    cons = [{"type": "ineq", "fun": (lambda j: (lambda p: p[i] - p[j]))(j)}
            for j in range(c) if j != i]
    return _opt_brier(c, i, True, cons, rng)


def inf_xi_numeric(c, rng, i=0, k=1):
    cons = [{"type": "ineq", "fun": lambda p: p[k] - p[i]}]
    return _opt_brier(c, i, False, cons, rng)


def run(rng):
    out = []

    # -------------------------------------------------- T1 / T2 / T2a / T6
    r = Result("T1", "sup over psi of sum-form Brier equals (c-1)/c")
    r2 = Result("T2", "inf over xi of sum-form Brier equals 1/2 for all c > 2")
    r2a = Result("T2a", "minimiser of T2 is the 1/2-tie configuration")
    r6 = Result("T6", "lambda*_PBS = (c-2)/(2c)")

    worst = {"sup": 0.0, "inf": 0.0, "lam": 0.0, "t": 0.0}
    for c in C_VALUES:
        s, _ = sup_psi_numeric(c, rng)
        f, x = inf_xi_numeric(c, rng)
        worst["sup"] = max(worst["sup"], abs(s - sup_psi_brier(c)))
        worst["inf"] = max(worst["inf"], abs(f - inf_xi_brier(c)))
        worst["lam"] = max(worst["lam"], abs((s - f) - lambda_star_brier(c)))
        worst["t"] = max(worst["t"], abs(x[0] - 0.5), abs(x[1] - 0.5))

    r.add("closed form matches optimiser for c in %s" % C_VALUES,
          worst["sup"] < 1e-9, "max deviation %.3e" % worst["sup"])
    r2.add("inf equals 1/2 independent of c", worst["inf"] < 1e-9,
           "max deviation %.3e" % worst["inf"])
    r2a.add("minimiser has p_true = p_winner = 1/2", worst["t"] < 1e-6,
            "max coordinate deviation %.3e" % worst["t"])
    r6.add("lambda* = sup - inf = (c-2)/(2c)", worst["lam"] < 1e-9,
           "max deviation %.3e" % worst["lam"])

    # independent grid cross-check of the reduced branch-2 problem
    grid_ok, grid_ev = True, []
    for c in GRID_C:
        t = np.linspace(1e-6, 0.5, GRID_N)
        g = (1 - t) ** 2 + t ** 2 + ((1 - 2 * t) ** 2) / (c - 2)
        j = int(np.argmin(g))
        grid_ok &= close(t[j], 0.5, 1e-6) and close(g[j], 0.5, 1e-9)
        grid_ev.append("c=%d argmin t=%.6f min=%.9f" % (c, t[j], g[j]))
    r2.add("independent %d-point grid agrees" % GRID_N, grid_ok, "; ".join(grid_ev))
    r2a.add("grid confirms t = 1/2 for every c tested", grid_ok, "; ".join(grid_ev))

    # c = 2 boundary
    r6.add("lambda* = 0 at c = 2 (recovers the c > 2 condition)",
           close(lambda_star_brier(2), 0.0), "lambda*(2) = %.1f" % lambda_star_brier(2))

    out += [r, r2, r2a, r6]

    # ----------------------------------------------------------- T3 / T4 / T7
    r3 = Result("T3", "sup over psi of log loss equals log c")
    r4 = Result("T4", "inf over xi of log loss equals log 2")
    r7 = Result("T7", "lambda*_PLL = log(c/2)")

    ok3 = ok4 = ok7 = True
    for c in [3, 5, 13, 50, 100]:
        # sup over psi: minimise p_i subject to p_i >= p_j  ->  p_i = 1/c
        ok3 &= close(sup_psi_log(c), -np.log(1.0 / c))
        # inf over xi: maximise p_i subject to some p_k >= p_i -> p_i = 1/2
        ok4 &= close(inf_xi_log(c), -np.log(0.5))
        ok7 &= close(sup_psi_log(c) - inf_xi_log(c), lambda_star_log(c))
    r3.add("log c for c in {3,5,13,50,100}", ok3, "analytic identity")
    r4.add("log 2, independent of c", ok4, "analytic identity")
    r7.add("lambda* = log c - log 2 = log(c/2)", ok7, "analytic identity")
    r7.add("lambda* = 0 at c = 2", close(lambda_star_log(2), 0.0), "log(1) = 0")

    # brute-force sanity: random draws never beat the analytic bounds
    viol3 = viol4 = 0
    for c in [3, 5, 13]:
        for _ in range(20000 if FULL else 4000):
            p = rng.dirichlet(np.ones(c))
            i = int(rng.integers(c))
            if p[i] >= p.max() - 1e-12:                  # in psi (convention A)
                if -np.log(p[i]) > sup_psi_log(c) + 1e-9: viol3 += 1
            if p[i] < p.max() - 1e-12:                   # in xi (convention A)
                if -np.log(p[i]) < inf_xi_log(c) - 1e-9: viol4 += 1
    r3.add("no random draw exceeds the bound", viol3 == 0, "%d violations" % viol3)
    r4.add("no random draw falls below the bound", viol4 == 0, "%d violations" % viol4)

    out += [r3, r4, r7]
    return out
