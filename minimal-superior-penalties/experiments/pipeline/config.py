"""Load experiments/config.yaml and derive a stable config hash.

The config hash is recorded in every results row (see the `outputs.
required_columns` list in config.yaml) so that any reported number can be
traced back to the exact pre-registered design that produced it.
"""
import hashlib
import subprocess
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash(cfg=None, path=CONFIG_PATH):
    """Hash of the raw config.yaml bytes -- simpler and more literal than
    re-serializing the parsed dict (which can hit unorderable-key issues
    when PyYAML's implicit bool/str resolution mixes key types), and it's
    exactly what "the pre-registered design" means: the file as written.
    """
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def git_commit():
    """Best-effort git commit hash; 'no-git' if this isn't a git repo yet.

    experiments/config.yaml's own header requires a git commit to freeze the
    design before any result is read. This repo is not a git repository as
    of this pipeline being written -- see the PREREGISTERED_AT note in
    run_experiments.py. Every results row still records whatever this
    returns, so the gap is traceable rather than silently papered over.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return "no-git"


if __name__ == "__main__":
    cfg = load_config()
    print("config_hash:", config_hash(cfg))
    print("git_commit:", git_commit())
    print("datasets:", [d["id"] for d in cfg["datasets"]])
    print("regimes:", [r["id"] for r in cfg["regimes"]])
