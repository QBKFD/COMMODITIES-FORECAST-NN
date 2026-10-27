"""Transformer encoder for prices and macro inputs with cross-attention.

Each stream is projected to ``d_model``, given sinusoidal positions and
passed through ``num_layers`` encoder blocks (self-attention and feed-forward
sublayers with residual connections and layer normalisation). The price
tokens then attend to the macro tokens. The last day's representation of the
price stream and of the cross-attention output feed a small dense head.
"""

from tensorflow.keras.layers import Concatenate, Dropout, Input
from tensorflow.keras.models import Model

from src.architectures.blocks import LastStep, compile_model, cross_block, embed, encoder_block, head


def build_model(config, lookback, n_price_features, n_macro_features):
    d_model = config.get("d_model", 32)
    heads = config.get("attention_heads", 4)
    layers = config.get("num_layers", 1)
    dropout = config.get("dropout", 0.2)
    ff_dim = config.get("ff_dim", 2 * d_model)

    price_input = Input(shape=(lookback, n_price_features), name="price_input")
    macro_input = Input(shape=(lookback, n_macro_features), name="macro_input")

    p = embed(price_input, d_model, lookback)
    m = embed(macro_input, d_model, lookback)
    for _ in range(layers):
        p = encoder_block(p, heads, d_model // heads, ff_dim, dropout)
        m = encoder_block(m, heads, d_model // heads, ff_dim, dropout)
    c = cross_block(p, m, heads, d_model // heads, ff_dim, dropout)

    x = Dropout(dropout)(Concatenate()([LastStep()(p), LastStep()(c)]))
    out = head(x, config.get("dense_units", 64), dropout, config.get("horizon", 1))
    return compile_model(Model([price_input, macro_input], out), config)
