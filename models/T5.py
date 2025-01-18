from optimum.quanto import QuantizedTransformersModel
from transformers import T5EncoderModel

class QuantizedT5(QuantizedTransformersModel):
    auto_class = T5EncoderModel
    auto_class.from_config = auto_class._from_config
