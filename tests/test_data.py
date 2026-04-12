from unittest.mock import Mock
import torch
from torch.utils.data import Dataset

from scripts.data import get_split, build_collate_fn

class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


def test_split_sizes():
    dataset = DummyDataset()

    original_len = len(dataset)
    ratio = 0.8
    expected_train_len = int(original_len * ratio)
    expected_test_len = original_len - expected_train_len

    train, test = get_split(dataset, train_sz=ratio, seed=42)

    assert len(train) == expected_train_len
    assert len(test) == expected_test_len
    assert len(train) + len(test) == len(dataset)


def test_split_deterministic():
    dataset = DummyDataset()

    some_seed = 42
    train1, test1 = get_split(dataset, seed=some_seed)
    train2, test2 = get_split(dataset, seed=some_seed)

    assert train1.indices == train2.indices
    assert test1.indices == test2.indices

def test_collate_fn_calls_processor():
    processor = Mock()
    processor.return_value = {
        "pixel_values": torch.randn(2, 3, 10, 10),
        "labels": torch.tensor([1, 2])
    }

    collate_fn = build_collate_fn(processor)

    batch = [
        ("img1", {"a": 1}),
        ("img2", {"a": 2}),
    ]

    output = collate_fn(batch)

    processor.assert_called_once()
    args, kwargs = processor.call_args

    assert "images" in kwargs
    assert "annotations" in kwargs
    assert kwargs["return_tensors"] == "pt"

    assert "pixel_values" in output
    assert "labels" in output
