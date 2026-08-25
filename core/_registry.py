from typing import Any, Callable, Dict


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def decorator(cls_or_fn: Callable) -> Callable:
            if name in self._items:
                raise KeyError(f"{self.name} '{name}' is already registered")
            self._items[name] = cls_or_fn
            return cls_or_fn

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._items:
            raise KeyError(f"{self.name} '{name}' is not registered")
        return self._items[name](**kwargs)
