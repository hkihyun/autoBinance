from .lstm import LSTMModel
from .tf import TransformerModel
MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "tf": TransformerModel,
}