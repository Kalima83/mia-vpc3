import os
from copy import deepcopy
from time import perf_counter
from typing import Callable

import torch

from scripts.evaluate import eval_on_validation
from scripts.utils import subset_dict


def train_one_epoch(model, device, optimizer, train_loader, max_grad_norm: float = 1.0):
    """
    Entrena el modelo 1 epoch y devuelve el valor promedio de loss observado.
    """
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)

        labels = [
            {k: v.to(device) for k, v in target.items()}
            for target in batch["labels"]
        ]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss / max(len(train_loader), 1)


def train(
    model,
    processor,
    device,
    optimizer,
    train_loader,
    val_loader,
    epochs,
    patience,
    save_dir,
    class_names: list[str],
    eval_score_threshold: float,
    eval_iou_match_threshold: float,
    eval_class_metrics: bool,
    logger=None,
    max_grad_norm: float = 1.0,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
):
    """
    Entrena el modelo y devuelve histórico de métricas escalares, por clase y artefactos de evaluación.
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "map": [],
        "map_50": [],
        "map_75": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "inference_ms_per_image": [],
        "peak_vram_mb": [],
    }

    per_class_history = {
        "per_class_ap": {class_name: [] for class_name in class_names},
        "per_class_mar_100": {class_name: [] for class_name in class_names},
        "per_class_precision": {class_name: [] for class_name in class_names},
        "per_class_recall": {class_name: [] for class_name in class_names},
        "per_class_f1": {class_name: [] for class_name in class_names},
    }

    names = {
        "train_loss": "Train Loss",
        "val_loss": "Valid Loss",
        "map": "mAP",
        "map_50": "mAP@50",
        "map_75": "mAP@75",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "inference_ms_per_image": "Infer ms/img",
        "peak_vram_mb": "VRAM MB",
        "train_elapsed": "Train time (s)",
        "val_elapsed": "Valid time (s)",
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_artifacts = None
    last_artifacts = None

    for epoch in range(epochs):
        train_elapsed_secs = perf_counter()
        cur_train_loss = train_one_epoch(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            max_grad_norm=max_grad_norm,
        )
        train_elapsed_secs = perf_counter() - train_elapsed_secs

        val_elapsed_secs = perf_counter()
        cur_metrics, cur_artifacts = eval_on_validation(
            model=model,
            processor=processor,
            device=device,
            val_loader=val_loader,
            class_names=class_names,
            score_threshold=eval_score_threshold,
            iou_match_threshold=eval_iou_match_threshold,
            class_metrics=eval_class_metrics,
        )
        val_elapsed_secs = perf_counter() - val_elapsed_secs

        cur_metrics["train_loss"] = cur_train_loss
        last_artifacts = deepcopy(cur_artifacts)

        for metric_name, metric_value in cur_metrics.items():
            history[metric_name].append(metric_value)

        for artifact_name, class_values in per_class_history.items():
            current_values = cur_artifacts.get(artifact_name, {})
            for class_name in class_values:
                class_values[class_name].append(current_values.get(class_name, float("nan")))

        epoch_metrics = dict(cur_metrics)
        epoch_metrics["train_elapsed"] = train_elapsed_secs
        epoch_metrics["val_elapsed"] = val_elapsed_secs
        epoch_metrics["epoch"] = epoch + 1

        for artifact_name in [
            "per_class_ap",
            "per_class_mar_100",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
        ]:
            prefix = artifact_name.replace("per_class_", "")
            for class_name, value in cur_artifacts.get(artifact_name, {}).items():
                epoch_metrics[f"{prefix}/{class_name}"] = value

        log_str = f"[{epoch+1:>2}/{epochs}] "
        log_str += " | ".join([
            f"{names[metric_name]}: {epoch_metrics[metric_name]:.4f}"
            for metric_name in [
                "train_loss",
                "val_loss",
                "map",
                "map_50",
                "map_75",
                "precision",
                "recall",
                "f1",
                "iou",
                "inference_ms_per_image",
                "peak_vram_mb",
                "train_elapsed",
                "val_elapsed",
            ]
        ])

        print(log_str)

        if logger is not None:
            logger.info(log_str)

        if metric_logger is not None:
            metric_logger(epoch_metrics)

        if cur_metrics["val_loss"] < best_val_loss:
            best_val_loss = cur_metrics["val_loss"]
            epochs_no_improve = 0
            best_artifacts = deepcopy(cur_artifacts)

            model_state = subset_dict(
                cur_metrics,
                [
                    "val_loss",
                    "map",
                    "map_50",
                    "map_75",
                    "precision",
                    "recall",
                    "f1",
                    "iou",
                    "inference_ms_per_image",
                    "peak_vram_mb",
                ],
            )
            model_state["epoch"] = epoch
            model_state["model_state_dict"] = model.state_dict()

            torch.save(
                model_state,
                os.path.join(save_dir, "best_model.pth"),
            )

            print("Mejor modelo guardado")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping activado")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))

    return {
        "history": history,
        "per_class_history": per_class_history,
        "best_artifacts": best_artifacts,
        "last_artifacts": last_artifacts,
    }
