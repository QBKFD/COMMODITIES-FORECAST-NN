"""GRU price encoder with attention over the macro inputs.

The price stream is a recurrent encoder. The macro stream is projected to
``d_model``, given positions and passed through a self-attention block. The
price states then attend to the macro states (cross-attention with residual
connection and feed-forward layer). The last day of the recurrent stream and
of the cross-attention output are concatenated for the dense head.

This is a recurrent model with attention over the macro inputs, not a
Temporal Fusion Transformer: there are no variable-selection networks,
gating or quantile outputs.
"""

from tensorflow.keras.layers import Concatenate, Dropout, Input, LayerNormalization, GRU
from tensorflow.keras.models import Model

from src.architectures.blocks import LastStep, compile_model, cross_block, embed, encoder_block, head


def build_model(config, lookback, n_price_features, n_macro_features):
    units = config.get("units", 64)
    d_model = config.get("d_model", 32)
    heads = config.get("attention_heads", 4)
    dropout = config.get("dropout", 0.2)

    price_input = Input(shape=(lookback, n_price_features), name="price_input")
    macro_input = Input(shape=(lookback, n_macro_features), name="macro_input")

    p = GRU(units, return_sequences=True)(price_input)
    p = Dropout(dropout)(p)
    p = GRU(d_model, return_sequences=True)(p)
    p = LayerNormalization()(p)

    m = encoder_block(embed(macro_input, d_model, lookback), heads, d_model // heads, 2 * d_model, dropout)
    c = cross_block(p, m, heads, d_model // heads, 2 * d_model, dropout)

    x = Dropout(dropout)(Concatenate()([LastStep()(p), LastStep()(c)]))
    out = head(x, config.get("dense_units", 64), dropout, config.get("horizon", 1))
    return compile_model(Model([price_input, macro_input], out), config)
