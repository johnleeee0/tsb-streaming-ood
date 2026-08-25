from typing import Any, Callable, Dict

from core._registry import Registry


DATASET_REGISTRY = Registry("dataset")


def register(name: str) -> Callable:
    return DATASET_REGISTRY.register(name)


def build(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return DATASET_REGISTRY.build(name, **params)
