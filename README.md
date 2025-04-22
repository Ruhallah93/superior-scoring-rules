# Superior Scoring Rules for Probabilistic Evaluation of Single-Label Multi-Class Classification Tasks

[Rouhollah Ahmadian](https://www.linkedin.com/in/ruhollah-ahmadian)<sup>1</sup> ,
[Mehdi Ghatee](https://aut.ac.ir/cv/2174/MEHDI-GHATEE?slc_lang=en&&cv=2174&mod=scv)<sup>1</sup>,
[Johan Wahlström](https://emps.exeter.ac.uk/computer-science/staff/cw840)<sup>2</sup><br>
<sup>1</sup>Amirkabir University of Technology, <sup>2</sup>University of Exeter, <sup>3</sup>University of Tehran

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

## About The Project
Evaluation metrics are critical in assessing the performance of probabilistic classification models. They influence tasks such as model selection, checkpointing, and early stopping. While widely used, traditional metrics like the Brier Score and Logarithmic Loss exhibit certain inconsistencies that can mislead the evaluation process. Specifically, these metrics may assign better scores to incorrect predictions (false positives or false negatives) compared to correct predictions (true positives or true negatives), leading to suboptimal model selection and evaluation.

To illustrate this inconsistency, consider the following scenario:
\begin{itemize}
\item 
True Label: $[0, 1, 0]$,
\item 
Predicted Vector A: $[0.33, 0.34, 0.33]$,
\item 
Predicted Vector B: $[0.51, 0.49, 0]$.
\end{itemize}
Vector $A$ represents a correct prediction since the argmax of $A$ matches the true label, whereas Vector $B$ represents an incorrect prediction because its argmax does not correspond to the true label. Intuitively, scoring rules should reward $A$ with a better score than $B$, as $A$ achieves accurate classification. However, traditional scoring rules may not align with this intuition.
The following table compares the Brier Score and Logarithmic Loss for these vectors:

| extbf{Vector} | \textbf{True Label (Y)} | \textbf{Predicted Probabilities (P)} | \textbf{Brier Score}                                                 | \textbf{Logarithmic Loss}                | \textbf{Prediction State} |
|---------------|-------------------------|--------------------------------------|----------------------------------------------------------------------|------------------------------------------|---------------------------|
| \textbf{A}    | $[0, 1, 0]$             | $[0.33, 0.34, 0.33]$                 | $\sum (Y_i - P_i)^2 = (0-0.33)^2 + (1-0.34)^2 + (0-0.33)^2 = 0.6534$ | $-\log(P_{true}) = -\log(0.34) = 0.4685$ | \textbf{Correct}          |
| \textbf{B}    | $[0, 1, 0]$             | $[0.51, 0.49, 0.00]$                 | $\sum (Y_i - P_i)^2 = (0-0.51)^2 + (1-0.49)^2 + (0-0.00)^2 = 0.5202$ | $-\log(P_{true}) = -\log(0.49) = 0.3098$ | \textbf{Incorrect}        |

As shown in the table, while $A$ is the correct prediction, its Brier Score (0.6534) is not better than $B$’s (0.5202). In addition, the Logarithmic Loss favors $B$ (0.3098) over $A$ (0.4685). Such outcomes contradict the principle that correct classifications should consistently be favored over incorrect ones.


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

1. Superior_Scoring_Rules.ipynb: It includes the implementation and analysis of the two proposed superior scoring rules.
2. superior_scoring_rules.py: It includes the implementation of Penalized Brier Score (PBS) and Penalized Logarithmic Loss (PLL).
2. /history: This folder contains images of statistical analysis. 
3. /hyperparameters-tuning: This folder includes the results of hyperparameter tuning.

