"""Model registry for MODBUS-2023 pipeline."""

from .catb import CatBoostModel
from .rf import RandomForestModel
from .xgb import XGBoostModel
from .et import ExtraTreesModel
from .lgbm import LightGBMModel
from .ftt import FTTransformerModel
from .base import BaseModel, TreeModel, DLModel

MODEL_REGISTRY = {
    "xgb": XGBoostModel,
    "cat": CatBoostModel,
    "rf": RandomForestModel,
    "et": ExtraTreesModel,
    "lgbm": LightGBMModel,
    "ftt": FTTransformerModel,
}


def get_model(name):
    """Get model class by name. Returns an uninitialized instance."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
