from copy import deepcopy
from typing import Any


def subset_dict(d: dict[Any, Any], subset_k: list[Any]) -> dict[Any, Any]:
    return {k: v for k, v in d.items() if k in subset_k}


def deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged
