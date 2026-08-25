from typing import Any, Callable, Dict

from core._registry import Registry


BACKBONE_REGISTRY = Registry("backbone")
OOD_REGISTRY = Registry("ood")


def register_backbone(name: str) -> Callable:
    return BACKBONE_REGISTRY.register(name)


def register_ood(name: str) -> Callable:
    return OOD_REGISTRY.register(name)


def build_backbone(name: str, params: Dict[str, Any]) -> Any:
    return BACKBONE_REGISTRY.build(name, **params)


def build_ood(name: str, params: Dict[str, Any]) -> Any:
    return OOD_REGISTRY.build(name, **params)
