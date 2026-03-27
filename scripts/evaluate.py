from time import perf_counter

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def eval_on_validation(
    model,
    processor,
    device,
    val_loader,
    class_names: list[str],
    score_threshold: float = 0.5,
    iou_match_threshold: float = 0.5,
    class_metrics: bool = True,
):
    """
    Evaluación sobre validation con métricas de entrenamiento, negocio y rendimiento.
    """

    def convert_gt_boxes_to_xyxy(
        gt_boxes: torch.Tensor,
        orig_size: torch.Tensor,
    ) -> torch.Tensor:
        gt_boxes = gt_boxes.clone()
        h, w = orig_size

        if gt_boxes.numel() == 0:
            return gt_boxes.reshape(0, 4)

        if gt_boxes.max() <= 1.0:
            cx, cy, bw, bh = gt_boxes.unbind(1)
            x1 = cx - 0.5 * bw
            y1 = cy - 0.5 * bh
            x2 = cx + 0.5 * bw
            y2 = cy + 0.5 * bh
            gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)

            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h

        return gt_boxes

    def empty_per_class_dict(default_value: float = float("nan")) -> dict[str, float]:
        return {class_name: default_value for class_name in class_names}

    def metric_tensor_to_per_class(
        classes_tensor: torch.Tensor | None,
        values_tensor: torch.Tensor | None,
    ) -> dict[str, float]:
        per_class = empty_per_class_dict()

        if classes_tensor is None or values_tensor is None:
            return per_class

        for class_idx, value in zip(classes_tensor.tolist(), values_tensor.tolist()):
            if 0 <= int(class_idx) < len(class_names):
                per_class[class_names[int(class_idx)]] = float(value)

        return per_class

    def update_confusion_matrix(
        confusion_matrix: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> None:
        background_idx = len(class_names)

        if len(pred_scores) > 0:
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        if len(pred_scores) > 0:
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

        matched_gt: set[int] = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
        else:
            ious = None

        for pred_idx in range(len(pred_boxes)):
            best_gt_idx = None
            best_iou = -1.0

            if ious is not None:
                ordered_gt = torch.argsort(ious[pred_idx], descending=True)
                for gt_idx in ordered_gt.tolist():
                    if gt_idx in matched_gt:
                        continue
                    best_gt_idx = gt_idx
                    best_iou = float(ious[pred_idx, gt_idx])
                    break

            pred_label = int(pred_labels[pred_idx].item())

            if best_gt_idx is not None and best_iou >= iou_match_threshold:
                gt_label = int(gt_labels[best_gt_idx].item())
                confusion_matrix[gt_label, pred_label] += 1
                matched_gt.add(best_gt_idx)
            else:
                confusion_matrix[background_idx, pred_label] += 1

        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                gt_label = int(gt_labels[gt_idx].item())
                confusion_matrix[gt_label, background_idx] += 1

    def confusion_to_scores(confusion_matrix: torch.Tensor) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        num_classes = len(class_names)
        tp_total = confusion_matrix.diag()[:num_classes].sum().item()
        predicted_total = confusion_matrix[:, :num_classes].sum().item()
        actual_total = confusion_matrix[:num_classes, :].sum().item()

        fp_total = predicted_total - tp_total
        fn_total = actual_total - tp_total

        precision = tp_total / predicted_total if predicted_total > 0 else 0.0
        recall = tp_total / actual_total if actual_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}

        for class_idx, class_name in enumerate(class_names):
            tp = confusion_matrix[class_idx, class_idx].item()
            predicted = confusion_matrix[:, class_idx].sum().item()
            actual = confusion_matrix[class_idx, :].sum().item()

            class_precision = tp / predicted if predicted > 0 else 0.0
            class_recall = tp / actual if actual > 0 else 0.0
            class_f1 = (
                2 * class_precision * class_recall / (class_precision + class_recall)
                if (class_precision + class_recall) > 0 else 0.0
            )

            per_class_precision[class_name] = class_precision
            per_class_recall[class_name] = class_recall
            per_class_f1[class_name] = class_f1

        scores = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        per_class_scores = {
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
        }

        return scores, per_class_scores

    model.eval()
    total_val_loss = 0.0
    total_iou = 0.0
    total_boxes = 0
    total_inference_time = 0.0
    total_inference_images = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
    )
    metric.reset()

    confusion_matrix = torch.zeros(
        (len(class_names) + 1, len(class_names) + 1),
        dtype=torch.int64,
    )

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)

            labels = [
                {k: v.to(device) for k, v in target.items()}
                for target in batch["labels"]
            ]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = perf_counter()
            outputs_pred = model(pixel_values=pixel_values)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_inference_time += perf_counter() - inference_start
            total_inference_images += pixel_values.shape[0]

            outputs_loss = model(pixel_values=pixel_values, labels=labels)
            total_val_loss += outputs_loss.loss.item()

            target_sizes = torch.stack([
                target["orig_size"] for target in batch["labels"]
            ]).to(device)

            results = processor.post_process_object_detection(
                outputs_pred,
                threshold=0.01,
                target_sizes=target_sizes,
            )

            preds = []
            targets = []

            for batch_idx, result in enumerate(results):
                pred_boxes = result["boxes"].cpu()
                pred_scores = result["scores"].cpu()
                pred_labels = result["labels"].cpu()

                gt_item = batch["labels"][batch_idx]
                gt_boxes = convert_gt_boxes_to_xyxy(
                    gt_item["boxes"].cpu(),
                    gt_item["orig_size"].cpu(),
                )
                gt_labels = gt_item["class_labels"].cpu()

                preds.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                })
                targets.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                })

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_ious, _ = ious.max(dim=1)
                    total_iou += max_ious.sum().item()
                    total_boxes += len(max_ious)

                update_confusion_matrix(
                    confusion_matrix=confusion_matrix,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )

            metric.update(preds, targets)

    metrics_result = metric.compute()
    scalar_scores, per_class_detection_scores = confusion_to_scores(confusion_matrix)

    inference_ms_per_image = (
        1000.0 * total_inference_time / total_inference_images
        if total_inference_images > 0 else 0.0
    )
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    metrics = {
        "val_loss": total_val_loss / max(len(val_loader), 1),
        "map": metrics_result["map"].item(),
        "map_50": metrics_result["map_50"].item(),
        "map_75": metrics_result["map_75"].item(),
        "iou": total_iou / max(total_boxes, 1),
        "precision": scalar_scores["precision"],
        "recall": scalar_scores["recall"],
        "f1": scalar_scores["f1"],
        "inference_ms_per_image": inference_ms_per_image,
        "peak_vram_mb": peak_vram_mb,
    }

    artifacts = {
        "per_class_ap": metric_tensor_to_per_class(
            metrics_result.get("classes"),
            metrics_result.get("map_per_class"),
        ),
        "per_class_mar_100": metric_tensor_to_per_class(
            metrics_result.get("classes"),
            metrics_result.get("mar_100_per_class"),
        ),
        "per_class_precision": per_class_detection_scores["precision"],
        "per_class_recall": per_class_detection_scores["recall"],
        "per_class_f1": per_class_detection_scores["f1"],
        "confusion_matrix": confusion_matrix.tolist(),
        "confusion_matrix_labels": class_names + ["background"],
        "score_threshold": score_threshold,
        "iou_match_threshold": iou_match_threshold,
    }

    return metrics, artifacts
