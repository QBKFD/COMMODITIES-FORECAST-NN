"""Stacked GRU with separate price and macro streams.

Two streams over the same 30-day window: prices/technicals and macro/market
variables. Each goes through GRU recurrent layers; the last day of each
stream is concatenated and passed to a small dense head.
"""

from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LayerNormalization, GRU
from tensorflow.keras.models import Model

from src.architectures.blocks import LastStep, compile_model, head


def build_model(config, lookback, n_price_features, n_macro_features):
    units = config.get("units", 64)
    dropout = config.get("dropout", 0.2)

    price_input = Input(shape=(lookback, n_price_features), name="price_input")
    macro_input = Input(shape=(lookback, n_macro_features), name="macro_input")

    p = GRU(units, return_sequences=True)(price_input)
    p = Dropout(dropout)(p)
    p = GRU(units, return_sequences=True)(p)
    p = LayerNormalization()(LastStep()(p))

    m = GRU(max(units // 2, 8), return_sequences=True)(macro_input)
    m = LayerNormalization()(LastStep()(m))

    x = Dropout(dropout)(Concatenate()([p, m]))
    out = head(x, config.get("dense_units", 64), dropout, config.get("horizon", 1))
    return compile_model(Model([price_input, macro_input], out), config)
