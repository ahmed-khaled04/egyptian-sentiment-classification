"""Single training entry point.

    python -m src.train --config configs/tfidf_lr.yaml
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from src.models import tfidf_lr

TRAINERS = {
    "tfidf_lr": tfidf_lr.train,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    result = TRAINERS[cfg["type"]](cfg)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(load_config(args.config))
    val = result["val"]
    print(f"val macro-F1={val['macro_f1']:.4f} accuracy={val['accuracy']:.4f}")


if __name__ == "__main__":
    main()
