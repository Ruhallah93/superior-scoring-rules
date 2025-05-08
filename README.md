# Superior Scoring Rules: Better Metrics for Probabilistic Evaluation ([arXiv Preprint](https://arxiv.org/pdf/2407.17697))

> 📊 PBS and PLL are superior evaluation metrics for probabilistic classifiers, fixing flaws in Brier Score (MSE) and Log Loss (Cross-Entropy). Strictly proper, consistent, and better for model selection, early stopping, and checkpointing.
## 🔍 Problem with Traditional Metrics  
Accuracy-based metrics (Accuracy, F1) treat all correct predictions equally, ignoring confidence. In high-stakes domains, confidence calibration is critical:

- 🧬 Cancer Diagnosis: 51% vs. 99% confidence in malignancy should not be treated differently.

- 🏥 ICU Triage & Mortality: Overconfident mispredictions risk patient safety.

- 🤖 Autonomous Vehicles: Decisions depend on uncertainty about obstacles.

- 💰 Financial Risk Modeling: Pricing and investment hinge on calibrated probabilities.

- 🔒 Security Threat Detection: High-confidence false negatives undermine defenses.

Thus, Accuracy or F1 Score alone is insufficient: they ignore the confidence of predictions.

## ⚠️ Limitations of MSE & Cross-Entropy

Mean Squared Error (Brier Score) and Cross-Entropy (Log Loss) are strictly proper scoring rules, rewarding calibration. However, they can still favor incorrect predictions over correct ones. Example: 

| Vector | True Label (Y) | Predicted Probabilities (P) | Brier Score | Log Loss | State |
|--------|----------------|-----------------------------|-------------|----------|-------|
| **`A`**  | `[0, 1, 0]`    | `[0.33, 0.34, 0.33]`        | 0.6534      | 0.4685   | ✅ Correct |
| **`B`**  | `[0, 1, 0]`    | `[0.51, 0.49, 0.00]`        | 0.5202      | 0.3098   | ❌ Incorrect |  

Both MSE and Log Loss favor B over A, contradicting the principle of rewarding correct predictions.

## 🎯 Our Solution: PBS & PLL  
To ensure correct predictions always receive better scores, we introduce a penalty term for misclassifications:

- ✅ **Penalized Brier Score (PBS)**

- ✅ **Penalized Logarithmic Loss (PLL)**

These metrics are both strictly proper and superior (never favor wrong over right).

##  Definitions 
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


## 🚀 Quick Start

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

### Early Stopping & Checkpointing
Use PBS/PLL instead of val_loss:
```python
class PBSCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['val_pbs'] = pbs(self.validation_data[1], self.model.predict(self.validation_data[0]))

model.fit(..., callbacks=[PBSCallback(),
    tf.keras.callbacks.EarlyStopping(monitor='val_pbs', patience=5, mode='min'),
    tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_pbs', save_best_only=True)
])
```

## 📂 Project Structure

Below is an overview of the main files and folders:

```
├── Superior_Scoring_Rules.ipynb   # Implementation & analysis  
├── superior_scoring_rules.py      # PBS & PLL functions  
├── README.md                      # This file  
├── history/                       # Statistical analysis plots  
└── hyperparameters-tuning/        # Tuning results  
```

## 📄 Paper & Citation

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

## 🤝 How to Contribute  
- 🐛 Report bugs via [Issues](https://github.com/Ruhallah93/superior-scoring-rules/issues).  
- 💡 Suggest improvements via Pull Requests.  
- 🌟 **Star the repo** if you find it useful!

