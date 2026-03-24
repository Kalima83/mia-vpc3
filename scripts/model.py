import torch
from transformers import DetrForObjectDetection, DetrImageProcessor


def get_model(model_name: str, cls_list: list[str]) -> tuple[DetrForObjectDetection, DetrImageProcessor]:
    """
    Función de utilidad para generar modelo y processor a partir de nombre de HuggingFace.
    """
    # Processor
    processor = DetrImageProcessor.from_pretrained(model_name)

    # Mapeo de clases (6 defectos)
    id2label = {idx:name for idx,name in enumerate(cls_list)}
    label2id = {v: k for k, v in id2label.items()}

    # Modelo adaptado
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    return model, processor