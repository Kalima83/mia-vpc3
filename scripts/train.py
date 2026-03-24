import os
import torch
from scripts.evaluate import eval_on_validation
from time import perf_counter

from scripts.utils import subset_dict

def train_one_epoch(model, device, optimizer, train_loader):
    """
    Entrena el modelo 1 epoch y devuelve el valor promedio de loss observado.
    """
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)

        labels = [
            {k: v.to(device) for k, v in t.items()}
            for t in batch["labels"]
        ]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / max(len(train_loader), 1)

    return avg_train_loss

def train(model, processor, device, optimizer, train_loader, val_loader, epochs, patience, save_dir, logger = None):
    """
    Entrena el modelo de acuerdo a la configuración ingresada y devuelve el histórico de métricas sobre train y validation.
    En caso de pasarse un logger, además de stdout también loggea el reporte de métricas por epoch.
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "map": [],
        "map_50": [],
        "iou": []
    }

    names = {
        "train_loss": "Train Loss",
        "val_loss": "Valid Loss",
        "map": "mAP",
        "map_50": "mAP@50",
        "iou": "IoU",
        "train_elapsed": "Train time (s)",
        "val_elapsed": "Valid time (s)"
    }

    # =========================
    # BEST MODEL + EARLY STOP
    # =========================
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):
        # entrenar 1 epoch
        train_elapsed_secs = perf_counter()
        cur_train_loss = train_one_epoch(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        train_elapsed_secs = perf_counter() - train_elapsed_secs

        # validar
        val_elapsed_secs = perf_counter()
        cur_metrics = eval_on_validation(
            model=model,
            processor=processor,
            device=device,
            val_loader=val_loader,
        )
        val_elapsed_secs = perf_counter() - val_elapsed_secs

        # agregado a history
        cur_metrics["train_loss"] = cur_train_loss

        for metric, metric_value in cur_metrics.items():
            history[metric].append(metric_value)

        # loggear
        cur_metrics["train_elapsed"] = train_elapsed_secs
        cur_metrics["val_elapsed"] = val_elapsed_secs

        # buildear str para metricas
        log_str = f"[{epoch+1:>2}/{epochs}] "
        log_str += " | ".join([
            f"{names[metric_name]}: {cur_metrics[metric_name]:.4f}"
            for metric_name 
            in ["train_loss", "val_loss", "map", "map_50", "iou", "train_elapsed", "val_elapsed"] # orden custom
        ])

        print(log_str) # a consola va siempre

        if logger is not None:
            logger.info(log_str)

        # -------- GUARDAR MEJOR MODELO --------
        if cur_metrics["val_loss"] < best_val_loss:
            best_val_loss = cur_metrics["val_loss"]
            epochs_no_improve = 0

            model_state = subset_dict(cur_metrics, ["val_loss", "map", "map_50", "iou"])
            model_state["epoch"] = epoch
            model_state["model_state_dict"] = model.state_dict()

            torch.save(
                model_state, 
                os.path.join(save_dir, "best_model.pth")
            )

            print("Mejor modelo guardado")

        else:
            epochs_no_improve += 1

        # -------- EARLY STOPPING --------
        if epochs_no_improve >= patience:
            print("Early stopping activado")
            break

    # =========================
    # GUARDADO FINAL
    # =========================
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))

    return history
        