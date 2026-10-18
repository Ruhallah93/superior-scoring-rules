"""Model-selection simulation from a cached per-epoch criterion stream.

config.yaml:
    selection:
      modes: [checkpointing, early_stopping]
      early_stopping_patience: 10
      min_delta: 0.0
      restore_best: true

Both modes are simulated OFFLINE against the full-budget training run --
"Early stopping must be SIMULATED offline from the cached stream, not run
live, or each lambda would need its own training run."
"""


def select_checkpoint(criterion_by_epoch):
    """Best single epoch over the whole training run (lower criterion is
    better -- see scoring.py, every selection criterion here is oriented
    that way, including macro-F1-direct baselines which are negated).
    """
    return min(criterion_by_epoch, key=criterion_by_epoch.get)


def select_early_stopping(criterion_by_epoch, patience=10, min_delta=0.0,
                           restore_best=True):
    """Walks epochs in order; stops once `patience` epochs pass with no
    improvement of at least `min_delta`. With restore_best (the config
    default), the selected epoch is the best one seen before stopping, not
    the stopping epoch itself.
    """
    epochs = sorted(criterion_by_epoch)
    best_epoch = epochs[0]
    best_crit = criterion_by_epoch[best_epoch]
    since_improvement = 0
    stop_epoch = epochs[-1]
    for e in epochs[1:]:
        c = criterion_by_epoch[e]
        if c < best_crit - min_delta:
            best_crit = c
            best_epoch = e
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                stop_epoch = e
                break
    return best_epoch if restore_best else stop_epoch


def select(mode, criterion_by_epoch, patience=10, min_delta=0.0, restore_best=True):
    if mode == "checkpointing":
        return select_checkpoint(criterion_by_epoch)
    elif mode == "early_stopping":
        return select_early_stopping(criterion_by_epoch, patience, min_delta, restore_best)
    raise ValueError(f"unknown selection mode {mode!r}")
