from .lstm import LSTMModel
from .tf import TransformerModel
MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "tfmodel": TransformerModel,
}