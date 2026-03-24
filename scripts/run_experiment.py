import json
import os
import logging
from yaml import safe_load

import torch
from torch.utils.data import DataLoader

from scripts.data import DeepPCBDataset, get_split, build_collate_fn
from scripts.model import get_model
from scripts.train import train
from scripts.config import Config


def run_experiment(config):
    # DATA
    dataset = DeepPCBDataset(
        config.data.root_dir, 
        os.path.join(config.data.root_dir, "trainval.txt")
    )

    train_dataset, val_dataset = get_split(
        dataset=dataset,
        train_sz=config.data.train_sz,
        seed=config.data.seed,
    )

    # MODEL & OPTIM
    model, processor = get_model(
        model_name=config.model_name,
        cls_list=dataset.classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.train.optim.lr, 
        weight_decay=config.train.optim.weight_decay,
    )

    # DATALOADERS
    collate_fn = build_collate_fn(processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_sz.train,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_sz.valid,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # LOGGER
    # TODO
    # FIXME
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # TRAIN
    history = train(
        model=model,
        processor=processor,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.train.trainer.epochs,
        patience=config.train.trainer.patience,
        save_dir=config.train.save_dir,
        logger=logger,
    )

    # LOG HISTORY + CONFIG
    with open(os.path.join(config.train.save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    with open(os.path.join(config.train.save_dir, "config.json"), "w") as f:
        json.dump(config, f)



if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]

    with open(config_path) as f:
        config_raw = safe_load(f)
    
    config = Config(**config_raw)

    run_experiment(config)