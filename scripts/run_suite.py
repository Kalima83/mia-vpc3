import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from yaml import safe_load

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import Config, SuiteConfig
from scripts.run_experiment import run_experiment
from scripts.utils import deep_merge_dict


def build_suite_summary(
    suite_name: str,
    suite_dir: str,
    experiment_results: list[dict[str, object]],
) -> dict[str, object]:
    best_result = None
    best_val_loss = float("inf")

    for result in experiment_results:
        summary = result["summary"]
        current_best = summary.get("best_val_loss")

        if current_best is None or not math.isfinite(current_best):
            continue

        if current_best < best_val_loss:
            best_val_loss = current_best
            best_result = summary

    return {
        "suite_name": suite_name,
        "suite_dir": suite_dir,
        "best_experiment_name": None if best_result is None else best_result["experiment_name"],
        "best_val_loss": None if best_result is None else best_result["best_val_loss"],
        "best_run_dir": None if best_result is None else best_result["run_dir"],
        "experiments": [result["summary"] for result in experiment_results],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corre una suite de experimentos con overrides por experimento.")
    parser.add_argument("suite_config_path", help="Path del archivo YAML con la suite.")
    parser.add_argument("--test", action="store_true", help="Reduce fuertemente el tamaño del train set para testear pipeline más rápido")

    args = parser.parse_args()

    with open(args.suite_config_path) as f:
        suite_config_raw = safe_load(f)

    suite_config = SuiteConfig(**suite_config_raw)

    suite_name = suite_config.suite_name or f"detr_deeppcb_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = os.path.join("runs", suite_name)
    wandb_root_dir = os.path.join(suite_dir, "wandb")

    os.makedirs(suite_dir, exist_ok=True)

    experiment_results = []

    for experiment in suite_config.experiments:
        merged_config_raw = deep_merge_dict(suite_config.base_config, experiment.overrides)
        merged_config_raw["experiment_name"] = experiment.name

        train_config = merged_config_raw.setdefault("train", {})
        train_config["save_dir"] = os.path.join(suite_dir, experiment.name)
        os.makedirs(train_config["save_dir"], exist_ok=True)

        wandb_config = merged_config_raw.get("wandb")
        if wandb_config is not None and wandb_config.get("enabled", False):
            wandb_config.setdefault("group", suite_name)
            wandb_config.setdefault("name", f"{suite_name}_{experiment.name}")
            wandb_config.setdefault("dir", os.path.join(wandb_root_dir, experiment.name))
            os.makedirs(wandb_config["dir"], exist_ok=True)

        config = Config(**merged_config_raw)

        print(f"\n===== Iniciando experimento: {experiment.name} =====")
        experiment_results.append(run_experiment(config, test_run=args.test))

    suite_summary = build_suite_summary(
        suite_name=suite_name,
        suite_dir=suite_dir,
        experiment_results=experiment_results,
    )

    with open(os.path.join(suite_dir, "suite_summary.json"), "w") as f:
        json.dump(suite_summary, f, indent=2)

    print(f"Suite guardada en: {suite_dir}")
    print(f"Mejor experimento: {suite_summary['best_experiment_name']}")
    print(f"Mejor val_loss: {suite_summary['best_val_loss']}")
