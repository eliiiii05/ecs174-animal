"""Config loading, seeding, checkpoints and results logging."""
import csv
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "results" / "results.csv"
RESULT_FIELDS = ["run_id", "model", "train_config", "eval_set", "metric", "value", "date", "git_commit"]


def resolve(path) -> Path:
    """Paths in configs are relative to the repo root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path, overrides=None) -> dict:
    """Load a YAML config. `overrides` is a list like ["epochs=1", "batch_size=64"]."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        cfg[key] = yaml.safe_load(value)
    cfg["_config_file"] = Path(path).stem
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "nogit"


def append_results(run_id, model, train_config, eval_set, metrics: dict) -> None:

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["run_id"], r["eval_set"], r["metric"])] = r
    date, commit = datetime.now().strftime("%Y-%m-%d %H:%M"), git_commit()
    for metric, value in metrics.items():
        rows[(run_id, eval_set, metric)] = {"run_id": run_id, "model": model, "train_config": train_config,
                                            "eval_set": eval_set, "metric": metric,
                                            "value": round(float(value), 5), "date": date, "git_commit": commit}
    tmp = RESULTS_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    tmp.replace(RESULTS_CSV)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)
