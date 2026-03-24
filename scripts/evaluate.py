import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def eval_on_validation(model, processor, device, val_loader):
    """
    Evaluación de modelo sobre set de validación. Calcula mAP, mAP@50, IoU y loss.
    """
    def to_cpu(x: list[dict[str, torch.Tensor]], keys: list[str], idx: int) -> dict[str, torch.Tensor]:
        return {
            k : x[idx][k].cpu() 
            for k in keys
        }

    model.eval()
    total_val_loss = 0.0

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.reset()


    total_iou = 0.0
    total_boxes = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)

            labels = [
                {k: v.to(device) for k, v in t.items()}
                for t in batch["labels"]
            ]

            # LOSS
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            # PREDICCIONES (sin labels)
            outputs_pred = model(pixel_values=pixel_values)

            # tamaños reales
            target_sizes = torch.stack([
                t["orig_size"] for t in batch["labels"]
            ]).to(device)

            results = processor.post_process_object_detection(
                outputs_pred,
                threshold=0.01,
                target_sizes=target_sizes
            )

            preds = []
            targets = []

            for i in range(len(results)):
                # append predictions
                pred_i = to_cpu(results, ["boxes","scores","labels"], i)
                preds.append(pred_i)

                pred_boxes = pred_i["boxes"]
                gpu_gt_i = batch["labels"][i]
                gt_boxes = gpu_gt_i["boxes"].clone()

                h, w = gpu_gt_i["orig_size"]

                # si están normalizados (0–1)
                if gt_boxes.max() <= 1.0:
                    # cxcywh → xyxy
                    cx, cy, bw, bh = gt_boxes.unbind(1)
                    x1 = cx - 0.5 * bw
                    y1 = cy - 0.5 * bh
                    x2 = cx + 0.5 * bw
                    y2 = cy + 0.5 * bh
                    gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)

                    # pasar a píxeles
                    gt_boxes[:, [0, 2]] *= w
                    gt_boxes[:, [1, 3]] *= h

                # append targets
                targets.append({
                    "boxes": gt_boxes.cpu(),
                    "labels": gpu_gt_i["class_labels"].cpu()
                })

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_ious, _ = ious.max(dim=1)

                    total_iou += max_ious.sum().item()
                    total_boxes += len(max_ious)

                # ===== IoU =====

                pred_boxes = pred_i["boxes"] # already CPU
                gt_boxes = gpu_gt_i["boxes"].cpu()

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:

                    ious = box_iou(pred_boxes, gt_boxes)
                    max_ious, _ = ious.max(dim=1)

                    total_iou += max_ious.sum().item()
                    total_boxes += len(max_ious)

            if any(len(p["boxes"]) > 0 for p in preds):
                metric.update(preds, targets)

    avg_val_loss = total_val_loss / max(len(val_loader), 1)
    avg_iou = total_iou / max(total_boxes, 1)

    metrics_result = metric.compute()
    map_score = metrics_result["map"].item()
    map_50 = metrics_result["map_50"].item()

    return {
        "val_loss": avg_val_loss,
        "map": map_score,
        "map_50": map_50,
        "iou": avg_iou
    }
