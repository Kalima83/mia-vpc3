from typing import Any


def subset_dict(d: dict[Any,Any], subset_k: list[Any]) -> dict[Any,Any]:
    return {k:v for k,v in d.items() if k in subset_k}