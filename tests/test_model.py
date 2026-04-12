from unittest.mock import patch, MagicMock

from scripts.model import get_model


@patch("scripts.model.DetrForObjectDetection.from_pretrained")
@patch("scripts.model.DetrImageProcessor.from_pretrained")
def test_get_model_basic(mock_processor_from_pretrained, mock_model_from_pretrained):
    mock_processor = MagicMock(name="processor")
    mock_model = MagicMock(name="model")

    mock_processor_from_pretrained.return_value = mock_processor
    mock_model_from_pretrained.return_value = mock_model

    cls_list = ["open", "short", "mousebite"]
    model_name = "facebook/detr-resnet-50"

    model, processor = get_model(model_name, cls_list)

    # returned objects
    assert model is mock_model
    assert processor is mock_processor

    # processor called correctly
    mock_processor_from_pretrained.assert_called_once_with(model_name)

    # expected mappings
    expected_id2label = {k:v for k,v in enumerate(cls_list)}
    expected_label2id = {v: k for k, v in expected_id2label.items()}

    # model called with correct arguments
    mock_model_from_pretrained.assert_called_once()
    _, kwargs = mock_model_from_pretrained.call_args

    assert kwargs["num_labels"] == 3
    assert kwargs["id2label"] == expected_id2label
    assert kwargs["label2id"] == expected_label2id
    assert kwargs["ignore_mismatched_sizes"] is True


@patch("scripts.model.DetrImageProcessor.from_pretrained")
def test_processor_failure_propagates(mock_processor_from_pretrained):
    error_msg = "load failed"
    mock_processor_from_pretrained.side_effect = RuntimeError(error_msg)

    try:
        get_model("bad-model", ["a"])
    except RuntimeError as e:
        assert error_msg in str(e)
    else:
        assert False, "Expected exception was not raised"