import numpy as np
from itertools import product
rng = np.random.default_rng(1)

def brier(p, i):
    y = np.zeros_like(p); y[i] = 1.0
    return float(np.sum((p - y) ** 2))

def pen_convA(p, i):
    """released code: penalised iff p_i < max(p).  Ties -> NOT penalised."""
    return 1.0 if p[i] < p.max() - 1e-12 else 0.0

def pen_convB(p, i):
    """single-valued decision rule: penalised iff argmax(p) != i."""
    return 1.0 if int(np.argmax(p)) != i else 0.0

def expected(S, p, Q, lam, pen):
    return sum(Q[i] * (brier(p, i) + lam * pen(p, i)) for i in range(len(Q)))

print("Q = (0.5, 0.3, 0.2), c = 3")
Q = np.array([0.5, 0.3, 0.2]); u = np.full(3, 1/3)
print(f"{'lam':>7} | {'E[S(Q)] A':>10} {'E[S(u)] A':>10} {'A proper?':>10} | "
      f"{'E[S(Q)] B':>10} {'E[S(u)] B':>10} {'B proper?':>10}")
for lam in [0.0, 0.05, 0.0933, 0.1667, 0.6667, 2.0]:
    qA = expected(None, Q, Q, lam, pen_convA); uA = expected(None, u, Q, lam, pen_convA)
    qB = expected(None, Q, Q, lam, pen_convB); uB = expected(None, u, Q, lam, pen_convB)
    print(f"{lam:>7.4f} | {qA:>10.5f} {uA:>10.5f} {'yes' if qA<uA else 'NO':>10} | "
          f"{qB:>10.5f} {uB:>10.5f} {'yes' if qB<uB else 'NO':>10}")

print("\nBroad search for convention-A violations (uniform vs truthful):")
print(f"{'c':>4} {'lam':>9} {'violations':>12} {'worst deficit':>15}")
for c in [3, 5, 13]:
    u = np.full(c, 1/c)
    for lam in [(c-2)/(2*c), (c-1)/c, float(c-1)]:
        bad = 0; worst = 0.0
        for _ in range(3000):
            Q = rng.dirichlet(np.ones(c))
            qA = expected(None, Q, Q, lam, pen_convA)
            uA = expected(None, u, Q, lam, pen_convA)
            if uA <= qA:
                bad += 1; worst = max(worst, qA - uA)
        print(f"{c:>4} {lam:>9.4f} {bad:>8}/3000 {worst:>15.5f}")

print("\nSame search under convention B (single-valued argmax):")
print(f"{'c':>4} {'lam':>9} {'violations':>12}")
for c in [3, 5, 13]:
    u = np.full(c, 1/c)
    for lam in [(c-2)/(2*c), (c-1)/c, float(c-1)]:
        bad = sum(1 for _ in range(3000)
                  for Q in [rng.dirichlet(np.ones(c))]
                  if expected(None, u, Q, lam, pen_convB)
                     <= expected(None, Q, Q, lam, pen_convB))
        print(f"{c:>4} {lam:>9.4f} {bad:>8}/3000")
