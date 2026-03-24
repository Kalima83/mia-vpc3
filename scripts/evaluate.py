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

    metric = MeanAveragePrecision()
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
            target_sizes = []
            for t in batch["labels"]:
                if "orig_size" in t:
                    target_sizes.append(t["orig_size"].cpu())
                else:
                    target_sizes.append((pixel_values.shape[2], pixel_values.shape[3]))

            results = processor.post_process_object_detection(
                outputs_pred,
                threshold=0.5,
                target_sizes=target_sizes
            )

            preds = []
            targets = []

            for i in range(len(results)):
                pred_i = to_cpu(results, ["boxes","scores","labels"], i)
                target_i = to_cpu(batch["labels"], ["boxes","labels"], i)

                preds.append(pred_i)
                targets.append(target_i)

                """
                preds.append({
                    "boxes": results[i]["boxes"].cpu(),
                    "scores": results[i]["scores"].cpu(),
                    "labels": results[i]["labels"].cpu()
                })

                targets.append({
                    "boxes": batch["labels"][i]["boxes"].cpu(),
                    "labels": batch["labels"][i]["labels"].cpu()
                })

                # ===== IoU =====
                pred_boxes = results[i]["boxes"].cpu()
                gt_boxes = batch["labels"][i]["boxes"].cpu()
                """
                
                # ===== IoU =====
                pred_boxes = pred_i["boxes"]
                gt_boxes = target_i["boxes"]

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_ious, _ = ious.max(dim=1)

                    total_iou += max_ious.sum().item()
                    total_boxes += len(max_ious)

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
