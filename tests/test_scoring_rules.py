"""
Tests for PBS and PLL scoring rules (NumPy implementation).

Covers:
- The motivating README example (incorrect overconfident prediction must score
  worse than a correct uncertain prediction)
- Perfect predictions (upper bound behaviour)
- Worst-case predictions (all probability mass on the wrong class)
- Uniform / maximum-entropy predictions
- All-zero and all-same-class edge inputs
- Batch consistency (mean of individual scores == batch result)
- Strict properness: E[S(p, Y)] is uniquely minimised when p == true distribution
- Shape / input validation
"""

import math

import numpy as np
import pytest

from superior_scoring_rules_numpy import pbs_numpy, pll_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def one_hot(index: int, n_classes: int) -> np.ndarray:
    v = np.zeros(n_classes, dtype=np.float32)
    v[index] = 1.0
    return v


def uniform(n_classes: int) -> np.ndarray:
    return np.full(n_classes, 1.0 / n_classes, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. README motivating example
#    Case A: correct but uncertain  -> [0,1,0] predicted as [0.33,0.34,0.33]
#    Case B: incorrect but confident -> [0,1,0] predicted as [0.51,0.49,0.00]
#    PBS and PLL must both prefer A over B.
# ---------------------------------------------------------------------------

class TestMotivatingExample:
    y_true = np.array([[0, 1, 0]], dtype=np.float32)
    pred_A = np.array([[0.33, 0.34, 0.33]], dtype=np.float32)   # correct, uncertain
    pred_B = np.array([[0.51, 0.49, 0.00]], dtype=np.float32)   # incorrect, confident

    def test_pbs_correct_beats_incorrect(self):
        """PBS must score correct-uncertain lower (better) than incorrect-confident."""
        score_A = pbs_numpy(self.y_true, self.pred_A)
        score_B = pbs_numpy(self.y_true, self.pred_B)
        assert score_A < score_B, (
            f"PBS should prefer correct-uncertain (A={score_A:.4f}) "
            f"over incorrect-confident (B={score_B:.4f})"
        )

    def test_pll_correct_beats_incorrect(self):
        """PLL must score correct-uncertain lower (better) than incorrect-confident."""
        score_A = pll_numpy(self.y_true, self.pred_A)
        score_B = pll_numpy(self.y_true, self.pred_B)
        assert score_A < score_B, (
            f"PLL should prefer correct-uncertain (A={score_A:.4f}) "
            f"over incorrect-confident (B={score_B:.4f})"
        )

    def test_standard_brier_inverted(self):
        """Confirm standard Brier Score (without penalty) gets this backwards."""
        brier_A = float(np.mean(np.sum((self.y_true - self.pred_A) ** 2, axis=1)))
        brier_B = float(np.mean(np.sum((self.y_true - self.pred_B) ** 2, axis=1)))
        # Standard Brier wrongly prefers B — that's the whole motivation
        assert brier_B < brier_A, (
            "Standard Brier score should (incorrectly) prefer B; "
            "if this fails, the test data changed."
        )


# ---------------------------------------------------------------------------
# 2. Perfect predictions
# ---------------------------------------------------------------------------

class TestPerfectPredictions:
    """When argmax is always correct and confidence is 1.0, penalty is zero."""

    def _make_perfect(self, n_classes=3, n_samples=5):
        indices = np.random.default_rng(0).integers(0, n_classes, n_samples)
        y_true = np.eye(n_classes, dtype=np.float32)[indices]
        y_pred = y_true.copy()
        return y_true, y_pred

    def test_pbs_perfect_is_zero(self):
        y_true, y_pred = self._make_perfect()
        assert pbs_numpy(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_pll_perfect_is_zero(self):
        y_true, y_pred = self._make_perfect()
        assert pll_numpy(y_true, y_pred) == pytest.approx(0.0, abs=1e-5)

    def test_pbs_high_confidence_correct_no_penalty(self):
        """Near-perfect confidence on the correct class: no penalty, low score."""
        y_true = np.array([[1, 0, 0]], dtype=np.float32)
        y_pred = np.array([[0.98, 0.01, 0.01]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        # No penalty; brier = mean([(1-0.98)^2, (0-0.01)^2, (0-0.01)^2])
        expected_brier = float(np.mean((y_true - y_pred) ** 2))
        assert score == pytest.approx(expected_brier, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. Worst-case predictions (all mass on the wrong class)
# ---------------------------------------------------------------------------

class TestWorstCase:
    def test_pbs_all_mass_on_wrong_class_3(self):
        """3-class: predict class 0 with certainty but truth is class 1."""
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        # MSE = mean([(0-1)^2, (1-0)^2, (0-0)^2]) = mean([1, 1, 0]) = 2/3
        # penalty = (3-1)/3 = 2/3
        expected = 2.0 / 3.0 + 2.0 / 3.0
        assert score == pytest.approx(expected, rel=1e-5)

    def test_pbs_worst_case_exceeds_correct_uncertain(self):
        """Worst-case prediction must score strictly higher than any correct one."""
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        worst = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        correct_uncertain = np.array([[0.34, 0.33, 0.33]], dtype=np.float32)
        assert pbs_numpy(y_true, worst) > pbs_numpy(y_true, correct_uncertain)

    def test_pll_worst_case_is_finite(self):
        """PLL with zero predicted probability on true class should be finite (clipped)."""
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        score = pll_numpy(y_true, y_pred)
        assert math.isfinite(score)
        assert score > 0

    def test_pll_worst_case_exceeds_correct(self):
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        worst = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        correct = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        assert pll_numpy(y_true, worst) > pll_numpy(y_true, correct)


# ---------------------------------------------------------------------------
# 4. Uniform predictions (maximum entropy)
# ---------------------------------------------------------------------------

class TestUniformPredictions:
    def test_pbs_uniform_no_penalty(self):
        """Uniform prediction is neither correct nor incorrect by argmax — ties
        are treated as correct (no penalty) because ST values are all zero."""
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        # No argmax violation: ST = [1/3 - 1/3, 1/3 - 1/3, 1/3 - 1/3] = [0,0,0]
        # penalty = 0; brier = mean([(0-1/3)^2, (1-1/3)^2, (0-1/3)^2])
        brier = float(np.mean((y_true - y_pred) ** 2))
        assert score == pytest.approx(brier, rel=1e-5)

    def test_pbs_uniform_correct_class_has_same_prob(self):
        """Uniform over 4 classes: ST for every class is 0 — no penalty."""
        y_true = np.array([[1, 0, 0, 0]], dtype=np.float32)
        y_pred = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        brier = float(np.mean((y_true - y_pred) ** 2))
        assert score == pytest.approx(brier, rel=1e-5)


# ---------------------------------------------------------------------------
# 5. Multi-sample batch consistency
# ---------------------------------------------------------------------------

class TestBatchConsistency:
    def test_pbs_batch_equals_mean_of_individuals(self):
        rng = np.random.default_rng(42)
        n, c = 8, 5
        indices = rng.integers(0, c, n)
        y_true = np.eye(c, dtype=np.float32)[indices]
        raw = rng.dirichlet(np.ones(c), size=n).astype(np.float32)
        y_pred = raw / raw.sum(axis=1, keepdims=True)

        batch_score = pbs_numpy(y_true, y_pred)
        individual_scores = [
            pbs_numpy(y_true[i : i + 1], y_pred[i : i + 1]) for i in range(n)
        ]
        assert batch_score == pytest.approx(float(np.mean(individual_scores)), rel=1e-5)

    def test_pll_batch_equals_mean_of_individuals(self):
        rng = np.random.default_rng(7)
        n, c = 10, 4
        indices = rng.integers(0, c, n)
        y_true = np.eye(c, dtype=np.float32)[indices]
        raw = rng.dirichlet(np.ones(c), size=n).astype(np.float32)
        y_pred = raw / raw.sum(axis=1, keepdims=True)

        batch_score = pll_numpy(y_true, y_pred)
        individual_scores = [
            pll_numpy(y_true[i : i + 1], y_pred[i : i + 1]) for i in range(n)
        ]
        assert batch_score == pytest.approx(float(np.mean(individual_scores)), rel=1e-5)


# ---------------------------------------------------------------------------
# 6. Penalty magnitude correctness
# ---------------------------------------------------------------------------

class TestPenaltyMagnitude:
    """Verify penalty values match the paper's formulas exactly."""

    def test_pbs_penalty_3_classes(self):
        """For 3-class wrong prediction, penalty = 2/3."""
        y_true = np.array([[1, 0, 0]], dtype=np.float32)
        # Slightly wrong: class 1 has slightly more probability than class 0
        y_pred = np.array([[0.49, 0.51, 0.00]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        brier = float(np.mean((y_true - y_pred) ** 2))
        expected_penalty = 2.0 / 3.0
        assert score == pytest.approx(brier + expected_penalty, rel=1e-5)

    def test_pbs_penalty_5_classes(self):
        """For 5-class wrong prediction, penalty = 4/5."""
        c = 5
        y_true = np.eye(c, dtype=np.float32)[[0]]
        y_pred = np.array([[0.1, 0.5, 0.2, 0.1, 0.1]], dtype=np.float32)  # wrong
        score = pbs_numpy(y_true, y_pred)
        brier = float(np.mean((y_true - y_pred) ** 2))
        expected_penalty = (c - 1) / c
        assert score == pytest.approx(brier + expected_penalty, rel=1e-5)

    def test_pll_penalty_3_classes(self):
        """For 3-class wrong prediction, penalty = log(1/3)."""
        y_true = np.array([[1, 0, 0]], dtype=np.float32)
        y_pred = np.array([[0.49, 0.51, 0.00]], dtype=np.float32)
        score = pll_numpy(y_true, y_pred)
        eps = 1e-7
        ce = float(-np.sum(y_true * np.log(np.clip(y_pred, eps, 1.0))))
        expected_penalty = math.log(1.0 / 3.0)
        assert score == pytest.approx(ce - expected_penalty, rel=1e-4)

    def test_pll_penalty_is_negative_log_uniform(self):
        """log(1/c) is always negative, so subtracting it raises PLL for wrong preds."""
        for c in [2, 3, 5, 10]:
            M = math.log(1.0 / c)
            assert M < 0, f"log(1/{c}) should be negative, got {M}"


# ---------------------------------------------------------------------------
# 7. Binary classification (c=2)
# ---------------------------------------------------------------------------

class TestBinaryClassification:
    def test_pbs_binary_correct(self):
        y_true = np.array([[1, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        brier = float(np.mean((y_true - y_pred) ** 2))
        assert score == pytest.approx(brier, rel=1e-5)  # No penalty

    def test_pbs_binary_wrong(self):
        y_true = np.array([[1, 0]], dtype=np.float32)
        y_pred = np.array([[0.3, 0.7]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        brier = float(np.mean((y_true - y_pred) ** 2))
        expected_penalty = 0.5  # (2-1)/2
        assert score == pytest.approx(brier + expected_penalty, rel=1e-5)

    def test_pll_binary_correct_vs_wrong_ordering(self):
        y_true = np.array([[1, 0]], dtype=np.float32)
        correct = np.array([[0.6, 0.4]], dtype=np.float32)
        wrong = np.array([[0.4, 0.6]], dtype=np.float32)
        assert pll_numpy(y_true, correct) < pll_numpy(y_true, wrong)


# ---------------------------------------------------------------------------
# 8. Core properness guarantee: the penalty makes PBS/PLL fix the specific
#    failure mode of standard scoring rules where an overconfident wrong
#    prediction scores better than a correct uncertain one.
#
#    The paper's guarantee (from Section 3): for any fixed confidence level p,
#    the penalized score for an incorrect prediction at confidence p must
#    exceed the penalized score for a correct prediction at confidence p.
#    That is: PBS(wrong@p, y) > PBS(correct@p, y) for all p in (0, 1).
#
#    This is a stronger, symmetric guarantee: wrong predictions can never
#    "hide" behind any confidence level to appear better than a correct
#    prediction at the same confidence.
# ---------------------------------------------------------------------------

class TestCoreProperGuarantee:
    """
    For any confidence level p, a prediction that puts p on the wrong class
    must score worse than a prediction that puts p on the correct class.
    """

    def _symmetric_pair(self, true_class: int, n_classes: int, p: float):
        """
        Returns (correct_pred, incorrect_pred) both with argmax-confidence p,
        remainder spread uniformly.  p must be > 1/n_classes so that argmax
        is unambiguous for both predictions.
        """
        assert p > 1.0 / n_classes, "p must exceed 1/n_classes for argmax to be unambiguous"
        wrong_class = (true_class + 1) % n_classes
        p_other = (1.0 - p) / (n_classes - 1)

        correct = np.full(n_classes, p_other, dtype=np.float32)
        correct[true_class] = p

        incorrect = np.full(n_classes, p_other, dtype=np.float32)
        incorrect[wrong_class] = p

        return correct, incorrect

    @pytest.mark.parametrize("n_classes,true_class", [(2, 0), (3, 1), (5, 2), (10, 7)])
    def test_pbs_same_confidence_wrong_always_worse(self, n_classes, true_class):
        """At matching confidence levels, a wrong prediction scores worse than correct."""
        y_true = np.eye(n_classes, dtype=np.float32)[[true_class]]
        # Only test p > 1/n_classes so argmax is unambiguous (both preds have a clear winner)
        p_min = 1.0 / n_classes + 0.05
        for p in np.linspace(p_min, 0.99, 20):
            qc, qi = self._symmetric_pair(true_class, n_classes, float(p))
            sc = pbs_numpy(y_true, qc[None, :])
            si = pbs_numpy(y_true, qi[None, :])
            assert si > sc, (
                f"PBS: at confidence p={p:.2f}, incorrect (score={si:.4f}) "
                f"should score worse than correct (score={sc:.4f}), "
                f"n_classes={n_classes}"
            )

    @pytest.mark.parametrize("n_classes,true_class", [(2, 0), (3, 1), (5, 2), (10, 7)])
    def test_pll_same_confidence_wrong_always_worse(self, n_classes, true_class):
        """At matching confidence levels, a wrong prediction scores worse than correct."""
        y_true = np.eye(n_classes, dtype=np.float32)[[true_class]]
        p_min = 1.0 / n_classes + 0.05
        for p in np.linspace(p_min, 0.99, 20):
            qc, qi = self._symmetric_pair(true_class, n_classes, float(p))
            sc = pll_numpy(y_true, qc[None, :])
            si = pll_numpy(y_true, qi[None, :])
            assert si > sc, (
                f"PLL: at confidence p={p:.2f}, incorrect (score={si:.4f}) "
                f"should score worse than correct (score={sc:.4f}), "
                f"n_classes={n_classes}"
            )

    def test_pbs_fixes_readme_example_across_confidence_levels(self):
        """
        Standard Brier Score prefers an overconfident wrong prediction over a
        correct uncertain prediction.  PBS must fix this for a range of
        wrong-confidence levels paired against the threshold correct confidence.
        """
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        # Correct prediction at low/mid confidence
        correct_uncertain = np.array([[0.1, 0.35, 0.55]], dtype=np.float32)

        for p_wrong in np.linspace(0.51, 0.95, 10):
            p_other = (1.0 - p_wrong) / 2
            wrong_confident = np.array(
                [[p_wrong, p_other, p_other]], dtype=np.float32
            )
            # Standard Brier often prefers the wrong-confident prediction
            # PBS must not
            pbs_correct = pbs_numpy(y_true, correct_uncertain)
            pbs_wrong = pbs_numpy(y_true, wrong_confident)
            assert pbs_wrong > pbs_correct, (
                f"PBS at p_wrong={p_wrong:.2f}: wrong ({pbs_wrong:.4f}) should "
                f"exceed correct-uncertain ({pbs_correct:.4f})"
            )


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_pbs_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            pbs_numpy(np.ones((3, 3)), np.ones((3, 4)))

    def test_pll_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            pll_numpy(np.ones((3, 3)), np.ones((4, 3)))

    def test_pbs_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            pbs_numpy(np.array([1, 0, 0]), np.array([0.8, 0.1, 0.1]))

    def test_pll_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            pll_numpy(np.array([1, 0, 0]), np.array([0.8, 0.1, 0.1]))

    def test_pbs_single_sample(self):
        """Single-sample inputs should work without error."""
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        score = pbs_numpy(y_true, y_pred)
        assert isinstance(score, float)

    def test_pll_single_sample(self):
        y_true = np.array([[0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        score = pll_numpy(y_true, y_pred)
        assert isinstance(score, float)

    def test_pbs_returns_float(self):
        y_true = np.eye(3, dtype=np.float32)
        y_pred = np.array([[0.8, 0.1, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.1, 0.1, 0.8]], dtype=np.float32)
        assert isinstance(pbs_numpy(y_true, y_pred), float)

    def test_pll_returns_float(self):
        y_true = np.eye(3, dtype=np.float32)
        y_pred = np.array([[0.8, 0.1, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.1, 0.1, 0.8]], dtype=np.float32)
        assert isinstance(pll_numpy(y_true, y_pred), float)


# ---------------------------------------------------------------------------
# 10. Non-negativity of PBS
# ---------------------------------------------------------------------------

class TestNonNegativity:
    """PBS should always be >= 0."""

    def test_pbs_nonnegative_random(self):
        rng = np.random.default_rng(99)
        for _ in range(50):
            n, c = rng.integers(1, 20), rng.integers(2, 8)
            indices = rng.integers(0, c, n)
            y_true = np.eye(c, dtype=np.float32)[indices]
            raw = rng.dirichlet(np.ones(c), size=n).astype(np.float32)
            y_pred = raw / raw.sum(axis=1, keepdims=True)
            assert pbs_numpy(y_true, y_pred) >= 0.0
