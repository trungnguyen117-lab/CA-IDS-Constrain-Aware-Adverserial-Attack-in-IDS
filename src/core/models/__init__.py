"""Models: BaseModel hierarchy + Registry + tree implementations."""

from .base import BaseModel, DLModel, MLModel
from .registry import Registry, build_model, get_registry
from .trees import (
    CatBoostModel,
    ExtraTreesModel,
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
)

# Register the standard tree targets into the global registry.
_R = get_registry()
_R.register("rf", RandomForestModel)
_R.register("et", ExtraTreesModel)
_R.register("xgb", XGBoostModel)
_R.register("cat", CatBoostModel)
_R.register("lgbm", LightGBMModel)

__all__ = [
    "BaseModel",
    "CatBoostModel",
    "DLModel",
    "ExtraTreesModel",
    "LightGBMModel",
    "MLModel",
    "RandomForestModel",
    "Registry",
    "XGBoostModel",
    "build_model",
    "get_registry",
]
