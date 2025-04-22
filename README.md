# Superior Scoring Rules: Better Metrics for Probabilistic Evaluation

> 📊 PBS and PLL are improved evaluation metrics for probabilistic classifiers, fixing flaws in Brier Score and Log Loss. Strictly proper, consistent, and better for model selection.

## 🔍 Problem with Traditional Metrics  
- ❌ Brier Score and Log Loss sometimes favor wrong predictions.  
- ❌ They can give better scores to incorrect models.

Evaluation metrics are critical in assessing the performance of probabilistic classification models. They influence tasks such as model selection, checkpointing, and early stopping. While widely used, traditional metrics like the Brier Score and Logarithmic Loss exhibit certain inconsistencies that can mislead the evaluation process. Specifically, these metrics may assign better scores to incorrect predictions (false positives or false negatives) compared to correct predictions (true positives or true negatives), leading to suboptimal model selection and evaluation.

To illustrate this inconsistency, consider the following scenario:  
- True Label: `[0, 1, 0]`,  
- Predicted Vector `A`: `[0.33, 0.34, 0.33]`,  
- Predicted Vector `B`: `[0.51, 0.49, 0]`.  

Vector `A` represents a correct prediction since the argmax of `A` matches the true label, whereas Vector `B` represents an incorrect prediction because its argmax does not correspond to the true label. Intuitively, scoring rules should reward `A` with a better score than `B`, as `A` achieves accurate classification. However, traditional scoring rules may not align with this intuition.  

**Comparison of Brier Score and Logarithmic Loss for Vectors `A` and `B`**  

| Vector | True Label (Y) | Predicted Probabilities (P) | Brier Score | Log Loss | State |
|--------|----------------|-----------------------------|-------------|----------|-------|
| **`A`**  | `[0, 1, 0]`    | `[0.33, 0.34, 0.33]`        | 0.6534      | 0.4685   | ✅ Correct |
| **`B`**  | `[0, 1, 0]`    | `[0.51, 0.49, 0.00]`        | 0.5202      | 0.3098   | ❌ Incorrect |  

As shown in the table, while `A` is the correct prediction, its Brier Score (0.6534) is not better than `B`’s (0.5202). In addition, the Logarithmic Loss favors `B` (0.3098) over `A` (0.4685). Such outcomes contradict the principle that correct classifications should consistently be favored over incorrect ones.

## 🎯 Our Solution: PBS & PLL  
- ✅ Fixes inconsistency by adding a penalty term.

To address this gap, this research introduces the **Penalized Brier Score (PBS)** and **Penalized Logarithmic Loss (PLL)**. These metrics integrate a penalty term for misclassifications, ensuring that:
- Correct predictions consistently receive better scores.
- Scoring rules align with the overarching goal of prioritizing accuracy in model evaluation.

The modified Brier Score with the penalty term, Penalized Brier Score (*PBS*), can be expressed as:

```math
S_{PBS}(q,i) = \sum_{i=1}^{c}(y_i-q_i)^2 + 
\begin{cases}
\frac{c-1}{c} & q \in \xi\\ 
0 & \text{otherwise}
\end{cases}
```

The modified Logarithmic Loss with the penalty term, Penalized Logarithmic Loss (*PLL*), can be expressed as:

```math
S_{PLL}(q,i) = - \sum_{i=1}^{c} y_i \log(p_i) - 
\begin{cases}
\log (\frac{1}{c}) & q \in \xi\\ 
0 & \text{otherwise}
\end{cases}
```

where:
- $y$ is the ground-truth vector
- $q$ is the predicted probability vector by a probabilistic classifier
- $c$ is the number of classes


## Code

```python
import tensorflow as tf
import math

# Penalized Brier Score (PBS)
def pbs(y, q):
    y = tf.cast(y, tf.float32)
    c = y.get_shape()[1]

    ST = tf.math.subtract(q, tf.reduce_sum(tf.where(y == 1, q, y), axis=1)[:, None])
    ST = tf.where(ST < 0, tf.constant(0, dtype=tf.float32), ST)
    payoff = tf.reduce_sum(tf.math.ceil(ST), axis=1)
    M = (c - 1) / (c)
    payoff = tf.where(payoff > 0, tf.constant(M, dtype=tf.float32), payoff)
    return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.square(tf.math.subtract(y, q)), axis=1) + payoff)

# Penalized Logarithmic Loss (PLL) 
def pll(y, q):
    y = tf.cast(y, tf.float32)
    c = y.get_shape()[1]

    ST = tf.math.subtract(q, tf.reduce_sum(tf.where(y == 1, q, y), axis=1)[:, None])
    ST = tf.where(ST < 0, tf.constant(0, dtype=tf.float32), ST)
    payoff = tf.reduce_sum(tf.math.ceil(ST), axis=1)
    M = math.log(1 / c)
    payoff = tf.where(payoff > 0, tf.constant(M, dtype=tf.float32), payoff)
    log_loss = tf.keras.losses.categorical_crossentropy(y, q)
    p_log_loss = tf.cast(log_loss, tf.float32) - payoff
    return tf.math.reduce_mean(p_log_loss)
```

## Files & Directories

1. [Superior_Scoring_Rules.ipynb](https://github.com/Ruhallah93/superior-scoring-rules/blob/main/Superior_Scoring_Rules.ipynb): It includes the implementation and analysis of the two proposed superior scoring rules.
2. [superior_scoring_rules.py](https://github.com/Ruhallah93/superior-scoring-rules/blob/main/superior_scoring_rules.py): It includes the implementation of Penalized Brier Score (PBS) and Penalized Logarithmic Loss (PLL).
2. [/history](https://github.com/Ruhallah93/superior-scoring-rules/tree/main/history): This folder contains images of statistical analysis. 
3. [/hyperparameters-tuning](https://github.com/Ruhallah93/superior-scoring-rules/tree/main/hyperparameters-tuning): This folder includes the results of hyperparameter tuning.

## Paper

[Superior scoring rules for probabilistic evaluation of single-label multi-class classification tasks](https://www.sciencedirect.com/science/article/abs/pii/S0888613X25000623)

[Rouhollah Ahmadian](https://scholar.google.com/citations?user=WwHM50MAAAAJ&hl=en&oi=ao)<sup>1</sup> ,
[Mehdi Ghatee](https://scholar.google.com/citations?user=b7lfEJwAAAAJ&hl=en&oi=ao)<sup>1</sup>,
[Johan Wahlström](https://scholar.google.com/citations?user=9rHhb5IAAAAJ&hl=en)<sup>2</sup><br>
<sup>1</sup>Amirkabir University of Technology, <sup>2</sup>University of Exeter

*If you use our method, please cite it by*:
```
@article{ahmadian2025superior,
  title={Superior scoring rules for probabilistic evaluation of single-label multi-class classification tasks},
  author={Ahmadian, Rouhollah and Ghatee, Mehdi and Wahlstr{\"o}m, Johan},
  journal={International Journal of Approximate Reasoning},
  pages={109421},
  year={2025},
  publisher={Elsevier}
}
```

## 🤝 How to Contribute  
- 🐛 Report bugs via [Issues](https://github.com/Ruhallah93/superior-scoring-rules/issues).  
- 💡 Suggest improvements via Pull Requests.  
- 🌟 **Star the repo** if you find it useful!

