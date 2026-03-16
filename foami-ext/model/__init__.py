from .xgb   import XGBModel
from .catb  import CatBoostModel
from .rf    import RFModel
from .lstm  import LSTMModel
from .resdnn import ResDNNModel

__all__ = ['XGBModel', 'CatBoostModel', 'RFModel', 'LSTMModel', 'ResDNNModel',
           'MODEL_REGISTRY']

# model_name → (ModelClass, model_type)
MODEL_REGISTRY = {
    'xgb':    (XGBModel,      'tree'),
    'cat':    (CatBoostModel, 'tree'),
    'rf':     (RFModel,       'tree'),
    'lstm':   (LSTMModel,     'dl'),
    'resdnn': (ResDNNModel,   'dl'),
}
