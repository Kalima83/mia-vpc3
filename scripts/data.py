import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split, Subset


class DeepPCBDataset(Dataset):
    """
    TODO: completar
    """
    def __init__(self, root_dir, split_file):
        self.root_dir = root_dir

        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f.readlines()]

        # 6 clases
        self.classes = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, ann_rel_path = self.samples[idx]

        img_path = os.path.join(
            self.root_dir,
            img_rel_path.replace(".jpg", "_test.jpg")
        )
        ann_path = os.path.join(self.root_dir, ann_rel_path)

        image = Image.open(img_path).convert("RGB")

        annotations = []

        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    xmin, ymin, xmax, ymax, class_id = parts

                    xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])
                    class_id = int(class_id)

                    width = xmax - xmin
                    height = ymax - ymin

                    # evitar cajas inválidas
                    if width < 1 or height < 1:
                        continue

                    annotations.append({
                        "bbox": [xmin, ymin, width, height],
                        "category_id": class_id,
                        "area": width * height,
                        "iscrowd": 0
                    })

        target = {
            "image_id": idx,
            "annotations": annotations
        }

        return image, target


def get_split(dataset: Dataset, train_sz: float = 0.8, seed: int = 42) -> list[Subset]:
    """
    Función de utilidad para realizar split train/test
    """
    # Fijar semilla para reproducibilidad
    torch.manual_seed(seed)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    return random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

def build_collate_fn(processor):
    """
    Función de utilidad para generar función de collation, utilizando procesor específico de un modelo.
    """
    def collate_fn(batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]

        encoding = processor(
            images=images,
            annotations=annotations,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoding["pixel_values"],
            "labels": encoding["labels"]
        }
    return collate_fn