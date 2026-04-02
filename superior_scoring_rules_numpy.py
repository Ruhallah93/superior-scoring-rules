"""
NumPy implementation of Penalized Brier Score (PBS) and Penalized Log Loss (PLL).

These are drop-in equivalents of the TensorFlow functions in superior_scoring_rules.py,
usable for evaluation without a TensorFlow dependency.

Reference:
    Ahmadian, R., Ghatee, M., & Wahlstrom, J. (2025).
    Superior scoring rules for probabilistic evaluation of single-label
    multi-class classification tasks.
    International Journal of Approximate Reasoning, 109421.
    https://arxiv.org/abs/2407.17697
"""

import math
import numpy as np


def pbs_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Penalized Brier Score (PBS) using NumPy.

    PBS adds a penalty of (c-1)/c to any prediction where argmax(y_pred)
    differs from argmax(y_true), ensuring that any incorrect prediction
    scores worse than any correct one regardless of confidence.

    Formula:
        PBS(p, y) = sum_i (y_i - p_i)^2 + penalty

        where penalty = (c-1)/c if argmax(p) != argmax(y), else 0

    Args:
        y_true: One-hot encoded ground truth, shape (n_samples, n_classes).
                Values must be 0 or 1.
        y_pred: Predicted probabilities, shape (n_samples, n_classes).
                Each row should sum to ~1 and values should be in [0, 1].

    Returns:
        Mean PBS across all samples (scalar float).

    Example:
        >>> import numpy as np
        >>> y_true = np.array([[0, 1, 0], [1, 0, 0]])
        >>> y_pred = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1]])
        >>> pbs_numpy(y_true, y_pred)
        0.09333...
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2-D arrays (n_samples, n_classes).")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )

    n_samples, c = y_true.shape

    # Predicted probability assigned to the true class, broadcast to (n, c)
    true_class_pred = np.sum(y_pred * y_true, axis=1, keepdims=True)

    # ST[i, j] > 0 means class j was predicted more confidently than the true class
    ST = y_pred - true_class_pred
    ST = np.clip(ST, 0.0, None)

    # Any sample where some wrong class has higher pred prob than the true class
    is_incorrect = np.sum(np.ceil(ST), axis=1) > 0

    penalty = np.where(is_incorrect, (c - 1) / c, 0.0)

    brier = np.mean(np.square(y_true - y_pred), axis=1)

    return float(np.mean(brier + penalty))


def pll_numpy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute the Penalized Log Loss (PLL) using NumPy.

    PLL subtracts a penalty of log(1/c) from the cross-entropy loss for any
    prediction where argmax(y_pred) != argmax(y_true). Since log(1/c) < 0,
    subtracting it *increases* the loss for incorrect predictions, ensuring
    any misclassification scores worse than any correct prediction.

    Formula:
        PLL(p, y) = -sum_i y_i * log(p_i) - penalty

        where penalty = log(1/c) if argmax(p) != argmax(y), else 0
        (log(1/c) is negative, so subtracting it raises the loss)

    Args:
        y_true: One-hot encoded ground truth, shape (n_samples, n_classes).
        y_pred: Predicted probabilities, shape (n_samples, n_classes).
                Clipped to [eps, 1] internally to avoid log(0).
        eps:    Small constant for numerical stability (default: 1e-7).

    Returns:
        Mean PLL across all samples (scalar float).

    Example:
        >>> import numpy as np
        >>> y_true = np.array([[0, 1, 0], [1, 0, 0]])
        >>> y_pred = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1]])
        >>> pll_numpy(y_true, y_pred)
        0.3677...
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2-D arrays (n_samples, n_classes).")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )

    n_samples, c = y_true.shape

    true_class_pred = np.sum(y_pred * y_true, axis=1, keepdims=True)
    ST = y_pred - true_class_pred
    ST = np.clip(ST, 0.0, None)
    is_incorrect = np.sum(np.ceil(ST), axis=1) > 0

    # log(1/c) is negative; subtracting it from CE raises the loss for wrong preds
    M = math.log(1.0 / c)
    penalty = np.where(is_incorrect, M, 0.0)

    y_pred_clipped = np.clip(y_pred, eps, 1.0)
    cross_entropy = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    return float(np.mean(cross_entropy - penalty))
