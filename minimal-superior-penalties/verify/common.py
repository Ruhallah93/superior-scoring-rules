"""Shared definitions for the verification suite.

Every check module imports from here so that a scoring rule or tie convention is
defined in exactly one place.
"""
import numpy as np

TOL = 1e-12

# ----------------------------------------------------------------- base rules
def brier_sum(p, i):
    """Sum-form Brier score.  This is source Definition 6 / Eq. (42)."""
    y = np.zeros(len(p)); y[i] = 1.0
    return float(np.sum((p - y) ** 2))

def brier_mean(p, i):
    """Mean-form Brier score.  This is what source Algorithm 2 line 3 uses."""
    return brier_sum(p, i) / len(p)

def logloss(p, i):
    """Natural-log logarithmic loss."""
    return float(-np.log(max(p[i], 1e-300)))

BASES = {"brier_sum": brier_sum, "brier_mean": brier_mean, "logloss": logloss}

# ----------------------------------------------------------- tie conventions
# Each returns True iff the prediction counts as CORRECT for true class i.

def correct_A(p, i):
    """Ties -> correct.  Symmetric.  What the released code implements
    (ceil(0) == 0 leaves a tied sample unpenalised)."""
    return bool(p[i] >= p.max() - TOL)

def correct_B(p, i):
    """Ties -> incorrect.  Symmetric.  Requires a strict argmax at i."""
    others = np.delete(p, i)
    return bool(p[i] > others.max() + TOL)

def correct_C(p, i):
    """Single-valued decision rule (lowest-index argmax).  Asymmetric in the
    class labels, but is a genuine decision rule d: simplex -> classes."""
    return bool(int(np.argmax(p)) == i)

CONVENTIONS = {"A": correct_A, "B": correct_B, "C": correct_C}
CONVENTION_NAMES = {
    "A": "ties to correct (released code)",
    "B": "ties to incorrect (symmetric)",
    "C": "single-valued argmax (lowest index)",
}

# -------------------------------------------------------------- the family
def s_lambda(p, i, lam, base, conv):
    return base(p, i) + (0.0 if CONVENTIONS[conv](p, i) else lam)

def expected_score(P, Q, lam, base, conv):
    """E_Q[S_lambda(P, .)] -- the paper's S(P,Q)."""
    return float(sum(Q[i] * s_lambda(P, i, lam, base, conv)
                     for i in range(len(Q))))

# ------------------------------------------------------- closed-form values
def sup_psi_brier(c):  return (c - 1) / c
def inf_xi_brier(c):   return 0.5
def sup_psi_log(c):    return np.log(c)
def inf_xi_log(c):     return np.log(2.0)
def lambda_star_brier(c): return (c - 2) / (2 * c)
def lambda_star_log(c):   return np.log(c / 2)

# published / implemented penalties
def lam_definition6(c): return (c - 1) / c
def lam_implemented_sum_equiv(c): return float(c - 1)
def lam_pll_published(c): return np.log(c)

# ------------------------------------------------------------- test harness
class Result:
    def __init__(self, ledger_id, description):
        self.ledger_id = ledger_id
        self.description = description
        self.checks = []
    def add(self, name, passed, evidence=""):
        self.checks.append({"name": name, "passed": bool(passed),
                            "evidence": str(evidence)})
        return passed
    @property
    def passed(self):
        return all(ch["passed"] for ch in self.checks)
    def to_dict(self):
        return {"ledger_id": self.ledger_id, "description": self.description,
                "passed": self.passed, "checks": self.checks}

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol
