from .lstm import LSTMModel
from .tft import TransformerModel
MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "tftmodel": TransformerModel,
}