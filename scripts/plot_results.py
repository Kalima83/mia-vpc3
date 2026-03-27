import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mia-vpc3-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def maybe_add_legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def save_confusion_matrix_plot(
    run_dir: Path,
    artifacts: dict[str, object],
    filename: str = "confusion_matrix_best.png",
) -> Path | None:
    confusion_matrix = artifacts.get("confusion_matrix")
    labels = artifacts.get("confusion_matrix_labels")

    if confusion_matrix is None or labels is None:
        return None

    output_path = run_dir / filename

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


def save_per_class_history_plot(
    run_dir: Path,
    run_name: str,
    per_class_history: dict[str, dict[str, list[float]]],
    metric_names: list[str],
    title: str,
    filename: str,
) -> Path | None:
    available_metrics = [
        metric_name
        for metric_name in metric_names
        if metric_name in per_class_history and per_class_history[metric_name]
    ]

    if not available_metrics:
        return None

    nrows = len(available_metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, available_metrics):
        class_histories = per_class_history[metric_name]
        max_len = max((len(values) for values in class_histories.values()), default=0)
        epochs = list(range(1, max_len + 1))

        for class_name, values in sorted(class_histories.items()):
            if not values:
                continue
            ax.plot(range(1, len(values) + 1), values, label=class_name, linewidth=2)

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        maybe_add_legend(ax, fontsize=8, ncol=2)

    fig.suptitle(f"{title} - {run_name}", fontsize=16)
    fig.tight_layout()
    output_path = run_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_per_class_snapshot_plot(
    run_dir: Path,
    run_name: str,
    artifacts: dict[str, object],
    metric_names: list[str],
    title: str,
    filename: str,
) -> Path | None:
    available_metrics = [
        metric_name
        for metric_name in metric_names
        if metric_name in artifacts and artifacts[metric_name]
    ]

    if not available_metrics:
        return None

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(16, 4 * len(available_metrics)))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, available_metrics):
        metric_values = artifacts[metric_name]
        labels = list(metric_values.keys())
        values = [metric_values[label] for label in labels]
        ax.barh(labels, values, color="#4c78a8")
        ax.invert_yaxis()
        ax.set_title(metric_name)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"{title} - {run_name}", fontsize=16)
    fig.tight_layout()
    output_path = run_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_suite_per_class_heatmap(
    suite_dir: Path,
    run_artifacts: dict[str, dict[str, object]],
    metric_name: str,
    filename: str,
) -> Path | None:
    rows = []
    row_labels = []
    class_labels = None

    for run_name, artifacts in sorted(run_artifacts.items()):
        metric_values = artifacts.get(metric_name)
        if not metric_values:
            continue

        if class_labels is None:
            class_labels = list(metric_values.keys())

        row_labels.append(run_name)
        rows.append([metric_values.get(class_name, float("nan")) for class_name in class_labels])

    if not rows or class_labels is None:
        return None

    data = np.array(rows, dtype=float)
    output_path = suite_dir / filename

    fig, ax = plt.subplots(figsize=(max(10, len(class_labels) * 1.2), max(6, len(row_labels) * 0.6)))
    image = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(metric_name)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if math.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def infer_summary(run_name: str, run_dir: Path, history: dict[str, list[float]]) -> dict[str, object]:
    val_loss = history.get("val_loss", [])
    best_idx = None
    best_val = float("inf")

    for idx, value in enumerate(val_loss):
        if math.isfinite(value) and value < best_val:
            best_val = value
            best_idx = idx

    summary: dict[str, object] = {
        "experiment_name": run_name,
        "run_dir": str(run_dir),
        "epochs_ran": len(val_loss),
    }

    if best_idx is not None:
        summary["best_epoch"] = best_idx + 1
        for metric_name, values in history.items():
            if best_idx < len(values):
                summary[f"best_{metric_name}"] = values[best_idx]

    final_idx = len(val_loss) - 1
    if final_idx >= 0:
        for metric_name, values in history.items():
            if final_idx < len(values):
                summary[f"final_{metric_name}"] = values[final_idx]

    return summary


def save_run_plot(run_dir: Path, run_name: str, history: dict[str, list[float]]) -> Path:
    epochs = list(range(1, len(history.get("val_loss", [])) + 1))
    output_path = run_dir / "training_metrics.png"

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    ax_loss, ax_map, ax_prf, ax_iou, ax_perf, ax_text = axes.flatten()

    if "train_loss" in history:
        ax_loss.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train_loss", linewidth=2)
    if "val_loss" in history:
        ax_loss.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="val_loss", linewidth=2)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Value")
    ax_loss.grid(True, alpha=0.3)
    maybe_add_legend(ax_loss)

    for metric_name in ["map", "map_50", "map_75"]:
        if metric_name in history:
            ax_map.plot(range(1, len(history[metric_name]) + 1), history[metric_name], label=metric_name, linewidth=2)
    ax_map.set_title("Detection Metrics")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("Score")
    ax_map.grid(True, alpha=0.3)
    maybe_add_legend(ax_map)

    for metric_name in ["precision", "recall", "f1"]:
        if metric_name in history:
            ax_prf.plot(range(1, len(history[metric_name]) + 1), history[metric_name], label=metric_name, linewidth=2)
    ax_prf.set_title("Precision / Recall / F1")
    ax_prf.set_xlabel("Epoch")
    ax_prf.set_ylabel("Score")
    ax_prf.grid(True, alpha=0.3)
    maybe_add_legend(ax_prf)

    if "iou" in history:
        ax_iou.plot(range(1, len(history["iou"]) + 1), history["iou"], label="iou", linewidth=2, color="#1b9e77")
    ax_iou.set_title("IoU")
    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("Score")
    ax_iou.grid(True, alpha=0.3)
    maybe_add_legend(ax_iou)

    for metric_name in ["inference_ms_per_image", "peak_vram_mb"]:
        if metric_name in history:
            ax_perf.plot(range(1, len(history[metric_name]) + 1), history[metric_name], label=metric_name, linewidth=2)
    ax_perf.set_title("Performance")
    ax_perf.set_xlabel("Epoch")
    ax_perf.set_ylabel("Value")
    ax_perf.grid(True, alpha=0.3)
    maybe_add_legend(ax_perf)

    summary = infer_summary(run_name, run_dir, history)
    ax_text.axis("off")
    lines = [
        f"Run: {run_name}",
        f"Epochs: {summary.get('epochs_ran', 0)}",
        f"Best epoch: {summary.get('best_epoch', '-')}",
        f"Best val_loss: {summary.get('best_val_loss', float('nan')):.4f}" if summary.get("best_val_loss") is not None else "Best val_loss: -",
        f"Best map: {summary.get('best_map', float('nan')):.4f}" if summary.get("best_map") is not None else "Best map: -",
        f"Best map_50: {summary.get('best_map_50', float('nan')):.4f}" if summary.get("best_map_50") is not None else "Best map_50: -",
        f"Best map_75: {summary.get('best_map_75', float('nan')):.4f}" if summary.get("best_map_75") is not None else "Best map_75: -",
        f"Best precision: {summary.get('best_precision', float('nan')):.4f}" if summary.get("best_precision") is not None else "Best precision: -",
        f"Best recall: {summary.get('best_recall', float('nan')):.4f}" if summary.get("best_recall") is not None else "Best recall: -",
        f"Best f1: {summary.get('best_f1', float('nan')):.4f}" if summary.get("best_f1") is not None else "Best f1: -",
        f"Best iou: {summary.get('best_iou', float('nan')):.4f}" if summary.get("best_iou") is not None else "Best iou: -",
        f"Best infer ms/img: {summary.get('best_inference_ms_per_image', float('nan')):.4f}" if summary.get("best_inference_ms_per_image") is not None else "Best infer ms/img: -",
        f"Best VRAM MB: {summary.get('best_peak_vram_mb', float('nan')):.4f}" if summary.get("best_peak_vram_mb") is not None else "Best VRAM MB: -",
    ]
    ax_text.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle(f"Training Summary - {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_suite_loss_plot(suite_dir: Path, histories: dict[str, dict[str, list[float]]]) -> Path:
    output_path = suite_dir / "comparison_loss_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for run_name, history in histories.items():
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        if "train_loss" in history:
            axes[0].plot(epochs, history["train_loss"], label=run_name, linewidth=2)
        if "val_loss" in history:
            axes[1].plot(epochs, history["val_loss"], label=run_name, linewidth=2)

    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        maybe_add_legend(ax, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_suite_detection_plot(suite_dir: Path, histories: dict[str, dict[str, list[float]]]) -> Path:
    output_path = suite_dir / "comparison_detection_curves.png"
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    metric_axes = {
        "map": axes[0, 0],
        "map_50": axes[0, 1],
        "map_75": axes[1, 0],
        "iou": axes[1, 1],
        "precision": axes[2, 0],
        "f1": axes[2, 1],
    }

    for run_name, history in histories.items():
        epochs = list(range(1, len(history.get("val_loss", [])) + 1))
        for metric_name, ax in metric_axes.items():
            if metric_name in history:
                ax.plot(epochs, history[metric_name], label=run_name, linewidth=2)

    for metric_name, ax in metric_axes.items():
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        maybe_add_legend(ax, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_suite_summary_plot(suite_dir: Path, summaries: list[dict[str, object]]) -> Path:
    output_path = suite_dir / "summary_best_metrics.png"
    metrics = [
        ("best_val_loss", True),
        ("best_map", False),
        ("best_map_50", False),
        ("best_map_75", False),
        ("best_precision", False),
        ("best_recall", False),
        ("best_f1", False),
        ("best_iou", False),
        ("best_inference_ms_per_image", True),
        ("best_peak_vram_mb", True),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    axes_flat = list(axes.flatten())

    for ax, (metric_name, ascending) in zip(axes_flat, metrics):
        pairs = []
        for summary in summaries:
            value = summary.get(metric_name)
            if value is None:
                continue
            pairs.append((summary["experiment_name"], value))

        pairs.sort(key=lambda item: item[1], reverse=not ascending)

        if not pairs:
            ax.axis("off")
            continue

        labels = [item[0] for item in pairs]
        values = [item[1] for item in pairs]
        ax.barh(labels, values, color="#4c78a8")
        ax.invert_yaxis()
        ax.set_title(metric_name)
        ax.grid(True, axis="x", alpha=0.3)

    for ax in axes_flat[len(metrics):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_suite_summary(suite_dir: Path, summaries: list[dict[str, object]]) -> dict[str, object]:
    best_summary = None
    best_val_loss = float("inf")

    for summary in summaries:
        value = summary.get("best_val_loss")
        if value is None or not math.isfinite(value):
            continue
        if value < best_val_loss:
            best_val_loss = value
            best_summary = summary

    return {
        "suite_name": suite_dir.name,
        "suite_dir": str(suite_dir),
        "best_experiment_name": None if best_summary is None else best_summary["experiment_name"],
        "best_val_loss": None if best_summary is None else best_summary["best_val_loss"],
        "best_run_dir": None if best_summary is None else best_summary["run_dir"],
        "experiments": summaries,
    }


def discover_suite_runs(suite_dir: Path) -> list[Path]:
    run_dirs = []
    for child in sorted(suite_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "wandb":
            continue
        if (child / "history.json").exists():
            run_dirs.append(child)
    return run_dirs


def plot_single_run(run_dir: Path) -> list[Path]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"No encontre history.json en {run_dir}")

    history = load_json(history_path)
    generated = [save_run_plot(run_dir, run_dir.name, history)]

    per_class_history_path = run_dir / "per_class_history.json"
    if per_class_history_path.exists():
        per_class_history = load_json(per_class_history_path)
        output = save_per_class_history_plot(
            run_dir,
            run_dir.name,
            per_class_history,
            ["per_class_ap", "per_class_mar_100"],
            "Per-class AP and MAR",
            "per_class_ap_mar_curves.png",
        )
        if output is not None:
            generated.append(output)

        output = save_per_class_history_plot(
            run_dir,
            run_dir.name,
            per_class_history,
            ["per_class_precision", "per_class_recall", "per_class_f1"],
            "Per-class Precision / Recall / F1",
            "per_class_prf_curves.png",
        )
        if output is not None:
            generated.append(output)

    best_artifacts_path = run_dir / "best_eval_artifacts.json"
    if best_artifacts_path.exists():
        best_artifacts = load_json(best_artifacts_path)
        output = save_per_class_snapshot_plot(
            run_dir,
            run_dir.name,
            best_artifacts,
            ["per_class_ap", "per_class_mar_100", "per_class_precision", "per_class_recall", "per_class_f1"],
            "Best Epoch Per-class Metrics",
            "per_class_best_snapshot.png",
        )
        if output is not None:
            generated.append(output)

        output = save_confusion_matrix_plot(run_dir, best_artifacts)
        if output is not None:
            generated.append(output)

    return generated


def plot_suite(suite_dir: Path) -> list[Path]:
    run_dirs = discover_suite_runs(suite_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No encontre corridas con history.json dentro de {suite_dir}")

    histories: dict[str, dict[str, list[float]]] = {}
    summaries: list[dict[str, object]] = []
    run_artifacts: dict[str, dict[str, object]] = {}
    generated_paths: list[Path] = []

    for run_dir in run_dirs:
        history = load_json(run_dir / "history.json")
        histories[run_dir.name] = history
        generated_paths.extend(plot_single_run(run_dir))

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summaries.append(load_json(summary_path))
        else:
            summary = infer_summary(run_dir.name, run_dir, history)
            summaries.append(summary)
            with summary_path.open("w") as f:
                json.dump(summary, f, indent=2)

        best_artifacts_path = run_dir / "best_eval_artifacts.json"
        if best_artifacts_path.exists():
            run_artifacts[run_dir.name] = load_json(best_artifacts_path)

    generated_paths.append(save_suite_loss_plot(suite_dir, histories))
    generated_paths.append(save_suite_detection_plot(suite_dir, histories))
    generated_paths.append(save_suite_summary_plot(suite_dir, summaries))

    for metric_name, filename in [
        ("per_class_ap", "suite_per_class_ap_heatmap.png"),
        ("per_class_precision", "suite_per_class_precision_heatmap.png"),
        ("per_class_recall", "suite_per_class_recall_heatmap.png"),
        ("per_class_f1", "suite_per_class_f1_heatmap.png"),
    ]:
        output = save_suite_per_class_heatmap(suite_dir, run_artifacts, metric_name, filename)
        if output is not None:
            generated_paths.append(output)

    suite_summary = build_suite_summary(suite_dir, summaries)
    suite_summary_path = suite_dir / "suite_summary.json"
    with suite_summary_path.open("w") as f:
        json.dump(suite_summary, f, indent=2)
    generated_paths.append(suite_summary_path)

    return generated_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grafica una corrida individual o una suite completa a partir de history.json.")
    parser.add_argument("target_dir", help="Carpeta de la corrida o de la suite.")

    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()

    if not target_dir.exists():
        raise FileNotFoundError(f"No existe {target_dir}")

    if (target_dir / "history.json").exists():
        generated = plot_single_run(target_dir)
    else:
        generated = plot_suite(target_dir)

    print("Archivos generados:")
    for path in generated:
        print(path)
