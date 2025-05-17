# Superior Scoring Rules: Better Metrics for Probabilistic Evaluation [![PyPI Version](https://img.shields.io/pypi/v/superior-scoring-rules)](https://pypi.org/project/superior-scoring-rules/) [![License](https://img.shields.io/pypi/l/superior-scoring-rules)](LICENSE)

> 📊 PBS and PLL are superior evaluation metrics for probabilistic classifiers, fixing flaws in Brier Score (MSE) and Log Loss (Cross-Entropy). Strictly proper, consistent, and better for model selection, early stopping, and checkpointing.

---

### Table of Contents

1. [Motivation](#motivation)
2. [Limitations of Traditional Metrics](#limitations-of-traditional-metrics)
3. [Penalized Scoring Rules](#penalized-scoring-rules)

   * [Definitions](#definitions)
   * [Formulas](#formulas)
   * [Implementation](#implementation)
4. [Quick Start](#quick-start)

   * [Installation](#installation)
   * [Basic Usage](#basic-usage)
   * [Callbacks for Early Stopping & Checkpointing](#callbacks-for-early-stopping--checkpointing)
5. [Project Structure](#project-structure)
6. [Paper & Citation](#paper--citation)
7. [Contributing](#contributing)
8. [License](#license)

---

## Motivation

In many high-stakes applications, **confidence calibration** is critical. Traditional accuracy-based metrics (Accuracy, F1) ignore prediction confidence. Consider:

* **Cancer Diagnosis**: Differentiating 51% vs. 99% confidence in malignancy
* **ICU Triage**: Overconfident mispredictions risk patient safety
* **Autonomous Vehicles**: Handling uncertainties about obstacles
* **Financial Risk Modeling**: Pricing and investment decisions
* **Security Threat Detection**: High-confidence false negatives

**Accuracy** or **F1** score alone cannot capture this nuance.

## Limitations of Traditional Metrics

While **Brier Score (MSE)** and **Log Loss (Cross-Entropy)** are strictly proper scoring rules, they can still favor incorrect, overconfident predictions over more calibrated, correct ones.

|  Case | True Class |     Prediction     | Brier Score | Log Loss |              Notes              |
| :---: | :--------: | :----------------: | :---------: | :------: | :-----------------------------: |
| **A** |  `[0,1,0]` | `[0.33,0.34,0.33]` |    0.6534   |  0.4685  |  ✅ Correct, but low confidence  |
| **B** |  `[0,1,0]` | `[0.51,0.49,0.00]` |    0.5202   |  0.3098  | ❌ Incorrect, but "better" score |

Traditional scores prefer **B** over **A**, violating the principle that correct predictions should always be rewarded.

## Penalized Scoring Rules

We introduce a penalty term that ensures **any** incorrect prediction is scored worse than any correct one.

### Definitions 
Let **y** be the one‑hot true vector, **p** the predicted probability vector, and **c** the number of classes. Define the set of predictions:
```math
\xi  = \{\,p \mid \arg\max p \neq \arg\max y\}\quad\text{(incorrect predictions)}
```

### Formulas
Then the **Penalized Brier Score (PBS)** is:

```math
S_{PBS}(p,i) = \sum_{i=1}^{c}(y_i-p_i)^2 + 
\begin{cases}
\frac{c-1}{c} & p \in \xi\\ 
0 & \text{otherwise}
\end{cases}
```

And the **Penalized Logarithmic Loss (PLL)** is:

```math
S_{PLL}(p,i) = - \sum_{i=1}^{c} y_i \log(p_i) - 
\begin{cases}
\log (\frac{1}{c}) & p \in \xi\\ 
0 & \text{otherwise}
\end{cases}
```

### Implementation

Penalized Brier Score (PBS)

```python
def pbs(y, q):
    """
    Computes Penalized Brier Score.
    
    Args:
        y_true: Ground truth (one-hot encoded), shape [batch_size, num_classes]
        y_pred: Predicted probabilities, shape [batch_size, num_classes]
        
    Returns:
        Mean PBS across batch
    """
    y = tf.cast(y, tf.float32)
    c = y.get_shape()[1]

    # Calculate the payoff term
    ST = tf.math.subtract(q, tf.reduce_sum(tf.where(y == 1, q, y), axis=1)[:, None])
    ST = tf.where(ST < 0, tf.constant(0, dtype=tf.float32), ST)
    payoff = tf.reduce_sum(tf.math.ceil(ST), axis=1)
    M = (c - 1) / (c)
    payoff = tf.where(payoff > 0, tf.constant(M, dtype=tf.float32), payoff)
    
    # Brier score + penalty
    brier = tf.math.reduce_mean(tf.math.square(tf.math.subtract(y, q)), axis=1)
    return tf.math.reduce_mean(brier + payoff)
```

Penalized Logarithmic Loss (PLL)

```python
def pll(y, q):
    """
    Computes Penalized Logarithmic Loss.
    
    Args:
        y_true: Ground truth (one-hot encoded)
        y_pred: Predicted probabilities
        
    Returns:
        Mean PLL across batch
    """
    y = tf.cast(y, tf.float32)
    c = y.get_shape()[1]

    # Calculate the payoff term
    ST = tf.math.subtract(q, tf.reduce_sum(tf.where(y == 1, q, y), axis=1)[:, None])
    ST = tf.where(ST < 0, tf.constant(0, dtype=tf.float32), ST)
    payoff = tf.reduce_sum(tf.math.ceil(ST), axis=1)
    M = math.log(1 / c)
    payoff = tf.where(payoff > 0, tf.constant(M, dtype=tf.float32), payoff)
    log_loss = tf.keras.losses.categorical_crossentropy(y, q)

    # Cross-entropy - penalty
    ce_loss = tf.cast(log_loss, tf.float32)
    return tf.math.reduce_mean(ce_loss - payoff)
```


## Quick Start

### Installation
Install via PyPI:

```bash
pip install superior-scoring-rules
```

### Basic Usage
```python
import tensorflow as tf
from superior_scoring_rules import pbs, pll

# Sample data (batch_size=3, num_classes=4)
y_true = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
y_pred = tf.constant([[0.9, 0.05, 0.05, 0], 
                     [0.1, 0.8, 0.05, 0.05],
                     [0.1, 0.1, 0.1, 0.7]])

print("PBS:", pbs(y_true, y_pred).numpy())
print("PLL:", pll(y_true, y_pred).numpy())
```

### Callbacks for Early Stopping & Checkpointing

```python
class PBSCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['val_pbs'] = pbs(self.validation_data[1],
                              self.model.predict(self.validation_data[0]))

model.fit(...,
    callbacks=[
        PBSCallback(),
        tf.keras.callbacks.EarlyStopping(monitor='val_pbs', patience=5, mode='min'),
        tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_pbs', save_best_only=True)
    ]
)
```

## Project Structure

Below is an overview of the main files and folders:

```
├── Superior_Scoring_Rules.ipynb   # Implementation & analysis  
├── superior_scoring_rules.py      # PBS & PLL functions  
├── README.md                      # This file  
├── history/                       # Statistical analysis plots  
└── hyperparameters-tuning/        # Tuning results  
```

## Paper & Citation

- [Superior scoring rules for probabilistic evaluation of single-label multi-class classification tasks](https://www.sciencedirect.com/science/article/abs/pii/S0888613X25000623)

- arXiv: [2407.17697](https://arxiv.org/pdf/2407.17697)

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

## Contributing

- 🐛 Report bugs via Issues

- 💡 Suggest improvements via Pull Requests

- ⭐️ Star the repository if you find it useful!

## License

This project is licensed under the [BSD License](LICENSE).

