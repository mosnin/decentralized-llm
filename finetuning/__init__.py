__all__ = ["FederatedTrainer", "FinetuneConfig", "LocalDataset"]


def __getattr__(name: str):
    if name in ("FederatedTrainer", "FinetuneConfig"):
        from .trainer import FederatedTrainer, FinetuneConfig

        return {"FederatedTrainer": FederatedTrainer, "FinetuneConfig": FinetuneConfig}[name]
    if name == "LocalDataset":
        from .dataset import LocalDataset

        return LocalDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
