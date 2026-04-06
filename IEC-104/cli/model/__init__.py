"""Model registry for FOAMI+ pipeline."""

from .catb import CatBoostModel
from .rf import RandomForestModel
from .lstm import LSTMModel
from .resdnn import ResDNNModel
from .surrogate_resdnn import SurrogateResDNNModel

MODEL_REGISTRY = {
    "cat": CatBoostModel,
    "rf": RandomForestModel,
    "lstm": LSTMModel,
    "resdnn": ResDNNModel,
    "surrogate_resdnn": SurrogateResDNNModel,
}


def get_model(name):
    """Get model class by name. Returns an uninitialized instance."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
