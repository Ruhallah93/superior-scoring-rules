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
This study introduces novel superior scoring rules called Penalized Brier Score (PBS) and Penalized Logarithmic Loss (PLL) to improve model evaluation for probabilistic classification.
Traditional scoring rules like Brier Score and Logarithmic Loss sometimes assign better score to misclassifications in comparison with correct classifications.
This discrepancy from the actual preference for rewarding correct classifications can lead to suboptimal model selection.
By integrating penalties for misclassifications, PBS and PLL modify traditional proper scoring rules to consistently assign higher scores to correct predictions.

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

