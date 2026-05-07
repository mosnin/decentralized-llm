from .config import NodeConfig

__all__ = ["Node", "NodeConfig"]


def __getattr__(name: str):
    if name == "Node":
        from .server import Node  # heavy deps (torch) loaded on demand

        return Node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
