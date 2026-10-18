import numpy as np
from scipy.optimize import minimize, linprog
rng = np.random.default_rng(0)

def brier_sum(p, i):
    y = np.zeros_like(p); y[i] = 1.0
    return float(np.sum((p - y) ** 2))

def inf_xi_brier(c, i=0, k=1):
    """min sum-form Brier over closure of xi (winner class k, k != i)."""
    def obj(p): return brier_sum(p, i)
    cons = [{'type': 'eq', 'fun': lambda p: p.sum() - 1.0},
            {'type': 'ineq', 'fun': lambda p: p[k] - p[i]}]
    best = None
    for _ in range(40):
        x0 = rng.dirichlet(np.ones(c))
        r = minimize(obj, x0, constraints=cons, bounds=[(0, 1)] * c,
                     method='SLSQP', options={'maxiter': 800, 'ftol': 1e-14})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return best.fun, best.x

def sup_psi_brier(c, i=0):
    """max sum-form Brier over closure of psi."""
    def negobj(p): return -brier_sum(p, i)
    cons = [{'type': 'eq', 'fun': lambda p: p.sum() - 1.0}] + \
           [{'type': 'ineq', 'fun': (lambda j: (lambda p: p[i] - p[j]))(j)}
            for j in range(c) if j != i]
    best = None
    for _ in range(40):
        x0 = rng.dirichlet(np.ones(c))
        r = minimize(negobj, x0, constraints=cons, bounds=[(0, 1)] * c,
                     method='SLSQP', options={'maxiter': 800, 'ftol': 1e-14})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return -best.fun, best.x

print("=" * 78)
print("T1/T2  Brier endpoints (sum form)")
print("=" * 78)
print(f"{'c':>4} {'sup_psi':>10} {'(c-1)/c':>10} {'inf_xi':>10} {'1/2':>6} "
      f"{'p_true*':>9} {'p_win*':>9} {'lam*':>9} {'(c-2)/2c':>9}")
rows_brier = []
for c in [3, 4, 5, 6, 8, 10, 12, 13, 20, 50, 100]:
    s, _ = sup_psi_brier(c)
    f, xstar = inf_xi_brier(c)
    lam = s - f
    print(f"{c:>4} {s:>10.6f} {(c-1)/c:>10.6f} {f:>10.6f} {0.5:>6.3f} "
          f"{xstar[0]:>9.6f} {xstar[1]:>9.6f} {lam:>9.6f} {(c-2)/(2*c):>9.6f}")
    rows_brier.append((c, s, f, lam))

max_sup_err = max(abs(s - (c - 1) / c) for c, s, f, l in rows_brier)
max_inf_err = max(abs(f - 0.5) for c, s, f, l in rows_brier)
max_lam_err = max(abs(l - (c - 2) / (2 * c)) for c, s, f, l in rows_brier)
print(f"\nmax |sup - (c-1)/c| = {max_sup_err:.3e}")
print(f"max |inf - 1/2|     = {max_inf_err:.3e}")
print(f"max |lam - (c-2)/2c| = {max_lam_err:.3e}")

# independent grid confirmation of the tie-configuration claim
print("\ngrid cross-check of inf_xi (t = m, remainder spread evenly):")
for c in [3, 5, 13, 50]:
    ts = np.linspace(1e-6, 0.5, 2000001)
    g = (1 - ts) ** 2 + ts ** 2 + ((1 - 2 * ts) ** 2) / (c - 2)
    j = int(np.argmin(g))
    print(f"  c={c:>3}  argmin t={ts[j]:.6f}  min={g[j]:.9f}")

print()
print("=" * 78)
print("T3/T4  Logarithmic endpoints (natural log)")
print("=" * 78)
for c in [3, 5, 13, 50]:
    print(f"  c={c:>3}  sup_psi = -log(1/c) = {np.log(c):.6f}   "
          f"inf_xi = log 2 = {np.log(2):.6f}   lam* = log(c/2) = {np.log(c/2):.6f}")

print()
print("=" * 78)
print("E1  Is S_PBS(Q,Q) = S_BS(Q,Q)?  (published Eq. 46)")
print("=" * 78)
for c, Q in [(3, np.array([0.5, 0.3, 0.2])), (3, np.array([0.4, 0.35, 0.25])),
             (4, np.array([0.7, 0.1, 0.1, 0.1]))]:
    lam = (c - 1) / c
    mode = int(np.argmax(Q))
    exp_bs = sum(Q[i] * brier_sum(Q, i) for i in range(c))
    exp_pbs = sum(Q[i] * (brier_sum(Q, i) + (lam if i != mode else 0.0))
                  for i in range(c))
    pred = exp_bs + lam * (1 - Q.max())
    print(f"  c={c} Q={np.round(Q,3)}  S_BS(Q,Q)={exp_bs:.6f}  "
          f"S_PBS(Q,Q)={exp_pbs:.6f}  gap={exp_pbs-exp_bs:.6f}  "
          f"lam(1-maxQ)={lam*(1-Q.max()):.6f}  match={np.isclose(exp_pbs, pred)}")

print()
print("=" * 78)
print("T8  Strict propriety of S_lambda (Monte Carlo, sum-form Brier base)")
print("=" * 78)
worst = {}
for c in [3, 5, 13]:
    for lam in [0.0, (c - 2) / (2 * c), (c - 1) / c, float(c - 1), 10.0]:
        bad = 0; margin = np.inf
        for _ in range(4000):
            Q = rng.dirichlet(np.ones(c)); P = rng.dirichlet(np.ones(c))
            mQ, mP = int(np.argmax(Q)), int(np.argmax(P))
            sQ = sum(Q[i] * (brier_sum(Q, i) + (lam if i != mQ else 0)) for i in range(c))
            sP = sum(Q[i] * (brier_sum(P, i) + (lam if i != mP else 0)) for i in range(c))
            if sP <= sQ + 1e-12: bad += 1
            margin = min(margin, sP - sQ)
        worst[(c, round(lam, 4))] = (bad, margin)
        print(f"  c={c:>3} lam={lam:>8.4f}  violations={bad:>4}/4000  min margin={margin:.3e}")

print()
print("=" * 78)
print("E3  Code audit: published implementation vs Definition 6")
print("=" * 78)
print(f"{'c':>4} {'Def6 lam':>10} {'impl lam (sum-equiv)':>22} {'lam*':>10} "
      f"{'Def6/lam*':>10} {'impl/lam*':>10} {'impl/range_psi':>15}")
for c in [3, 5, 10, 12, 13, 20]:
    lam_def6 = (c - 1) / c
    lam_impl = float(c - 1)          # mean-form base + (c-1)/c, rescaled by c
    lam_star = (c - 2) / (2 * c)
    print(f"{c:>4} {lam_def6:>10.4f} {lam_impl:>22.4f} {lam_star:>10.4f} "
          f"{lam_def6/lam_star:>10.2f} {lam_impl/lam_star:>10.2f} "
          f"{lam_impl/((c-1)/c):>15.2f}")

print()
print("=" * 78)
print("E5  Table 1 log base")
print("=" * 78)
for name, pt in [("A (correct)", 0.34), ("B (incorrect)", 0.49)]:
    print(f"  {name:>14}  published={0.4685 if pt==0.34 else 0.3098:.4f}  "
          f"-log10={-np.log10(pt):.4f}  -ln={-np.log(pt):.4f}")
A = np.array([0.33, 0.34, 0.33]); B = np.array([0.51, 0.49, 0.0])
print(f"  Brier A: published=0.6534  sum-form={brier_sum(A,1):.4f}  mean-form={brier_sum(A,1)/3:.4f}")
print(f"  Brier B: published=0.5202  sum-form={brier_sum(B,1):.4f}  mean-form={brier_sum(B,1)/3:.4f}")
