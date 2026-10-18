"""Dataset loading, splits, and regime application, matching config.yaml's
`datasets`, `regimes`, and `regime_reporting.label_noise` sections.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as T

DATA_ROOT = None  # set by caller; torchvision downloads here


class NoisyLabelDataset(Dataset):
    """Wraps a dataset, applying symmetric uniform label noise to a fixed,
    seeded subset of indices. `regime_reporting.label_noise`:
        type: symmetric_uniform, applied_to: train_split_only
    """

    def __init__(self, base, classes, noise_rate, seed):
        self.base = base
        self.classes = classes
        n = len(base)
        rng = np.random.default_rng(seed)
        self.noisy_mask = rng.random(n) < noise_rate
        # symmetric uniform: replace with a uniformly random OTHER class
        self.replacement = rng.integers(0, classes - 1, size=n)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.noisy_mask[idx]:
            r = self.replacement[idx]
            y = r if r < y else r + 1  # uniform over the c-1 other classes
        return x, y


def _mnist(root, train, augment=False):
    tfs = [T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]
    return torchvision.datasets.MNIST(root=root, train=train, download=True,
                                       transform=T.Compose(tfs))


def _cifar10(root, train, augment=False):
    tfs = []
    if augment:
        tfs += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    tfs += [T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    return torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                         transform=T.Compose(tfs))


def _emnist_balanced(root, train, augment=False):
    # AMENDMENT 1, revised. `augment` is accepted
    # for signature parity with the other loaders but unused -- no dataset
    # in this study's config requests augmentation for a grayscale,
    # MNIST-scale dataset, matching _mnist above.
    #
    # NOTE: older torchvision releases had a well-documented EMNIST image
    # orientation bug (images come out transposed relative to their labels
    # unless corrected). Recent torchvision versions fix this internally: if
    # a smoke-test run shows implausibly low accuracy on this dataset
    # specifically (not mnist/cifar10), check for that before assuming
    # anything else is wrong.
    tfs = [T.ToTensor(), T.Normalize((0.1751,), (0.3332,))]
    return torchvision.datasets.EMNIST(root=root, split="balanced", train=train,
                                        download=True, transform=T.Compose(tfs))


LOADERS = {"mnist": _mnist, "cifar10": _cifar10, "emnist_balanced": _emnist_balanced}


def build_splits(dataset_cfg, regime_cfg, seed, root):
    """Returns (train_ds, val_ds, test_ds) for one (dataset, regime, seed)
    training run, with train_fraction and label_noise from the regime
    applied to the train split only, per config.yaml.
    """
    did = dataset_cfg["id"]
    classes = dataset_cfg["classes"]
    split = dataset_cfg["split"]
    augment = "augmentation" in dataset_cfg

    full_train = LOADERS[did](root, train=True, augment=augment)
    test_ds = LOADERS[did](root, train=False, augment=False)

    n_train, n_val = split["train"], split["val"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(full_train))
    train_idx, val_idx = perm[:n_train], perm[n_train:n_train + n_val]

    train_ds = Subset(full_train, train_idx.tolist())
    # validation must not use train-time augmentation; rebuild without it
    val_base = LOADERS[did](root, train=True, augment=False)
    val_ds = Subset(val_base, val_idx.tolist())

    frac = regime_cfg["train_fraction"]
    if frac < 1.0:
        k = int(len(train_ds) * frac)
        sub_idx = rng.permutation(len(train_ds))[:k]
        train_ds = Subset(train_ds, sub_idx.tolist())

    noise = regime_cfg.get("label_noise", 0.0)
    if noise > 0:
        train_ds = NoisyLabelDataset(train_ds, classes, noise, seed)

    return train_ds, val_ds, test_ds
