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


def run_experiment(config: Config, test_run: bool = False):
    # verify save dir exists
    assert os.path.exists(config.train.save_dir), "save_dir from config file does not exist"

    # DATA
    if not os.path.exists(config.data.root_dir):
        os.system("git clone https://github.com/tangsanli5201/DeepPCB.git")
        print("Downloaded dataset")
    else:
        print("Data path exists, skipped download")

    dataset = DeepPCBDataset(
        config.data.root_dir, 
        os.path.join(config.data.root_dir, "trainval.txt")
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
            seed=5
        )
        print("New train size is", len(train_dataset))

    print("Data formatted and split")
    
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

    print("Model and optimizer done")

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

    print("Dataloaders done")

    # LOGGER
    logging.basicConfig(
        filename=os.path.join(config.train.save_dir, "run.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger()


    print("Logger set")

    # TRAIN
    print("Starting training")

    history = train(
        model=model,
        processor=processor,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.train.trainer.epochs if not test_run else 2,
        patience=config.train.trainer.patience,
        save_dir=config.train.save_dir,
        logger=logger,
    )

    print("Training done")

    # LOG HISTORY + CONFIG
    with open(os.path.join(config.train.save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    with open(os.path.join(config.train.save_dir, "config.json"), "w") as f:
        json.dump(config.model_dump(), f)

    print("Saved history and config jsons")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corre un experimento con la configuración provista.")
    parser.add_argument("config_path", help="Path del archivo de configuración.")
    parser.add_argument("--use_colab", action="store_true", help="Usar Google Colab") # default False
    parser.add_argument("--test", action="store_true", help="Reduce fuertemente el tamaño del train set para testear pipeline más rápido")
    
    args = parser.parse_args()
    config_path = args.config_path

    if args.use_colab:
        print("Using Colab")
        from google.colab import drive
        drive.mount('/content/drive')

    with open(config_path) as f:
        config_raw = safe_load(f)

    config = Config(**config_raw)

    run_experiment(config, test_run = args.test)