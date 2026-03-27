import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

from yaml import safe_load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import Config
from scripts.data import DeepPCBDataset, build_collate_fn, get_split
from scripts.model import get_model
from scripts.train import train


def build_optimizer(model, config: Config) -> torch.optim.Optimizer:
    lr_backbone = config.train.optim.lr_backbone

    if lr_backbone is None:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.train.optim.lr,
            weight_decay=config.train.optim.weight_decay,
        )

    main_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" in name:
            backbone_params.append(param)
        else:
            main_params.append(param)

    param_groups = []

    if main_params:
        param_groups.append({
            "params": main_params,
            "lr": config.train.optim.lr,
        })

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": lr_backbone,
        })

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config.train.optim.weight_decay,
    )


def get_run_logger(save_dir: str, experiment_name: str) -> logging.Logger:
    logger_name = f"mia_vpc3.{experiment_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(os.path.join(save_dir, "run.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    return logger


def close_run_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def init_wandb_run(config: Config, default_name: str):
    wandb_config = config.wandb

    if (
        wandb_config is None
        or not wandb_config.enabled
        or wandb_config.mode == "disabled"
    ):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb no esta disponible; se continua sin logging remoto")
        return None

    wandb_dir = wandb_config.dir or config.train.save_dir
    os.makedirs(wandb_dir, exist_ok=True)

    init_kwargs = {
        "project": wandb_config.project,
        "name": wandb_config.name or default_name,
        "config": config.model_dump(mode="json"),
        "dir": wandb_dir,
        "mode": wandb_config.mode,
        "reinit": True,
    }

    if wandb_config.entity is not None:
        init_kwargs["entity"] = wandb_config.entity

    if wandb_config.group is not None:
        init_kwargs["group"] = wandb_config.group

    if wandb_config.tags:
        init_kwargs["tags"] = wandb_config.tags

    return wandb.init(**init_kwargs)


def save_history_csv(history: dict[str, list[float]], save_dir: str) -> None:
    csv_path = os.path.join(save_dir, "history.csv")
    metric_names = list(history.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *metric_names])

        for idx in range(len(history["train_loss"])):
            writer.writerow([
                idx + 1,
                *[history[metric_name][idx] for metric_name in metric_names],
            ])


def save_per_class_history_json(per_class_history: dict[str, dict[str, list[float]]], save_dir: str) -> None:
    with open(os.path.join(save_dir, "per_class_history.json"), "w") as f:
        json.dump(per_class_history, f, indent=2)


def save_eval_artifacts(artifacts: dict[str, object] | None, save_dir: str, filename: str) -> None:
    if artifacts is None:
        return

    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(artifacts, f, indent=2)


def save_confusion_matrix_plot(
    artifacts: dict[str, object] | None,
    save_dir: str,
    filename: str,
) -> str | None:
    if artifacts is None:
        return None

    confusion_matrix = artifacts.get("confusion_matrix")
    labels = artifacts.get("confusion_matrix_labels")

    if confusion_matrix is None or labels is None:
        return None

    output_path = os.path.join(save_dir, filename)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for row_idx, row in enumerate(confusion_matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def flatten_nested_metrics(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}/{key}": value
        for key, value in values.items()
    }


def get_best_epoch(history: dict[str, list[float]]) -> int | None:
    best_idx = None
    best_val_loss = float("inf")

    for idx, val_loss in enumerate(history["val_loss"]):
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_idx = idx

    return best_idx


def build_summary(config: Config, history: dict[str, list[float]]) -> dict[str, object]:
    best_idx = get_best_epoch(history)
    final_idx = len(history["val_loss"]) - 1 if history["val_loss"] else None

    summary: dict[str, object] = {
        "experiment_name": config.experiment_name,
        "run_dir": config.train.save_dir,
        "epochs_ran": len(history["val_loss"]),
    }

    if best_idx is not None:
        summary["best_epoch"] = best_idx + 1
        for metric_name, values in history.items():
            summary[f"best_{metric_name}"] = values[best_idx]
    else:
        summary["best_epoch"] = None

    if final_idx is not None:
        for metric_name, values in history.items():
            summary[f"final_{metric_name}"] = values[final_idx]

    return summary


def run_experiment(config: Config, test_run: bool = False):
    os.makedirs(config.train.save_dir, exist_ok=True)
    experiment_name = config.experiment_name or os.path.basename(config.train.save_dir)
    config = config.model_copy(update={"experiment_name": experiment_name})

    if not os.path.exists(config.data.root_dir):
        os.system("git clone https://github.com/tangsanli5201/DeepPCB.git")
        print("Downloaded dataset")
    else:
        print("Data path exists, skipped download")

    dataset = DeepPCBDataset(
        config.data.root_dir,
        os.path.join(config.data.root_dir, "trainval.txt"),
    )

    train_dataset, val_dataset = get_split(
        dataset=dataset,
        train_sz=config.data.train_sz,
        seed=config.data.seed,
    )

    if test_run:
        print("Test run -> reducing train size")
        print("Old train size was", len(train_dataset))
        train_dataset, _ = get_split(
            dataset=train_dataset,
            train_sz=0.1,
            seed=5,
        )
        print("New train size is", len(train_dataset))

    print("Data formatted and split")

    model, processor = get_model(
        model_name=config.model_name,
        cls_list=dataset.classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, config)

    print("Model and optimizer done")

    collate_fn = build_collate_fn(processor)
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_sz.train,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_sz.valid,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=use_pin_memory,
    )

    print("Dataloaders done")

    logger = get_run_logger(config.train.save_dir, experiment_name)
    wandb_run = init_wandb_run(config, experiment_name)

    print("Logger set")
    print("Starting training")
    try:
        train_result = train(
            model=model,
            processor=processor,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.train.trainer.epochs if not test_run else 2,
            patience=config.train.trainer.patience,
            save_dir=config.train.save_dir,
            class_names=dataset.classes,
            eval_score_threshold=config.eval.score_threshold,
            eval_iou_match_threshold=config.eval.iou_match_threshold,
            eval_class_metrics=config.eval.class_metrics,
            logger=logger,
            max_grad_norm=config.train.trainer.max_grad_norm,
            metric_logger=wandb_run.log if wandb_run is not None else None,
        )

        print("Training done")

        history = train_result["history"]
        per_class_history = train_result["per_class_history"]
        best_artifacts = train_result["best_artifacts"]
        last_artifacts = train_result["last_artifacts"]

        save_history_csv(history, config.train.save_dir)
        save_per_class_history_json(per_class_history, config.train.save_dir)
        save_eval_artifacts(best_artifacts, config.train.save_dir, "best_eval_artifacts.json")
        save_eval_artifacts(last_artifacts, config.train.save_dir, "last_eval_artifacts.json")
        confusion_matrix_path = save_confusion_matrix_plot(
            best_artifacts,
            config.train.save_dir,
            "confusion_matrix_best.png",
        )

        with open(os.path.join(config.train.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        with open(os.path.join(config.train.save_dir, "config.json"), "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)

        summary = build_summary(config, history)

        with open(os.path.join(config.train.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if wandb_run is not None:
            wandb_run.summary.update(summary)
            if best_artifacts is not None:
                for prefix in [
                    "per_class_ap",
                    "per_class_mar_100",
                    "per_class_precision",
                    "per_class_recall",
                    "per_class_f1",
                ]:
                    wandb_run.summary.update(flatten_nested_metrics(prefix, best_artifacts.get(prefix, {})))
            if confusion_matrix_path is not None:
                import wandb
                wandb_run.log({"best_confusion_matrix": wandb.Image(confusion_matrix_path)})

        print("Saved history and config jsons")

        return {
            "history": history,
            "summary": summary,
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()

        close_run_logger(logger)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corre un experimento con la configuración provista.")
    parser.add_argument("config_path", help="Path del archivo de configuración.")
    parser.add_argument("--use_colab", action="store_true", help="Usar Google Colab")
    parser.add_argument("--test", action="store_true", help="Reduce fuertemente el tamaño del train set para testear pipeline más rápido")

    args = parser.parse_args()

    if args.use_colab:
        print("Using Colab")
        from google.colab import drive
        drive.mount('/content/drive')

    with open(args.config_path) as f:
        config_raw = safe_load(f)

    config = Config(**config_raw)
    run_experiment(config, test_run=args.test)
