from copy import deepcopy

from scripts.utils import subset_dict, deep_merge_dict


# ---------- subset_dict ----------

def test_subset_dict_basic():
    d = {"a": 1, "b": 2, "c": 3}
    result = subset_dict(d, ["a", "c"])

    assert result == {"a": 1, "c": 3}


def test_subset_dict_missing_keys_ignored():
    d = {"a": 1, "b": 2}
    result = subset_dict(d, ["a", "z"])

    assert result == {"a": 1}


def test_subset_dict_empty_subset():
    d = {"a": 1}
    result = subset_dict(d, [])

    assert result == {}


def test_subset_dict_does_not_mutate_input():
    d = {"a": 1, "b": 2}
    original = deepcopy(d)

    _ = subset_dict(d, ["a"])
    assert d == original


# ---------- deep_merge_dict ----------

def test_deep_merge_basic_override():
    base = {"a": 1, "b": 2}
    overrides = {"b": 3}

    result = deep_merge_dict(base, overrides)

    assert result == {"a": 1, "b": 3}


def test_deep_merge_nested_dicts():
    base = {"a": {"x": 1, "y": 2}}
    overrides = {"a": {"y": 99}}

    result = deep_merge_dict(base, overrides)

    assert result == {"a": {"x": 1, "y": 99}}


def test_deep_merge_add_new_keys():
    base = {"a": 1}
    overrides = {"b": 2}

    result = deep_merge_dict(base, overrides)

    assert result == {"a": 1, "b": 2}


def test_deep_merge_override_non_dict_with_dict():
    base = {"a": 1}
    overrides = {"a": {"x": 2}}

    result = deep_merge_dict(base, overrides)

    assert result == {"a": {"x": 2}}


def test_deep_merge_override_dict_with_non_dict():
    base = {"a": {"x": 1}}
    overrides = {"a": 5}

    result = deep_merge_dict(base, overrides)

    assert result == {"a": 5}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    overrides = {"a": {"x": 2}}

    base_copy = deepcopy(base)
    overrides_copy = deepcopy(overrides)

    _ = deep_merge_dict(base, overrides)

    assert base == base_copy
    assert overrides == overrides_copy


def test_deep_merge_deepcopy_behavior():
    base = {"a": {"x": [1, 2]}}
    overrides = {"a": {"x": [3]}}

    result = deep_merge_dict(base, overrides)

    # mutate result, ensure originals unaffected
    result["a"]["x"].append(999)

    assert base["a"]["x"] == [1, 2]
    assert overrides["a"]["x"] == [3]


def test_deep_merge_empty_overrides():
    base = {"a": 1}
    overrides = {}

    result = deep_merge_dict(base, overrides)

    assert result == base


def test_deep_merge_empty_base():
    base = {}
    overrides = {"a": 1}

    result = deep_merge_dict(base, overrides)

    assert result == {"a": 1}
