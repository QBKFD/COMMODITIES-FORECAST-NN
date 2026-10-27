import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class PositionalEncoding(Layer):
    """Adds fixed sinusoidal position information (Vaswani et al., 2017).

    Self-attention is permutation invariant: without this the model cannot
    tell the most recent day from the oldest one in the window.
    """

    def __init__(self, length: int, dim: int, **kwargs):
        super().__init__(**kwargs)
        pos = np.arange(length)[:, None]
        i = np.arange(dim)[None, :]
        angle = pos / np.power(10000.0, (2 * (i // 2)) / dim)
        table = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
        self.table = tf.constant(table[None, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.table


def _encode(inputs, lookback, d_model, heads, key_dim, dropout):
    """Project to d_model, add positions, one self-attention block with a residual."""
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(lookback, d_model)(x)
    attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)
    attn = Dropout(dropout)(attn)
    x = LayerNormalization()(x + attn)
    ff = Dense(d_model, activation="relu")(x)
    return LayerNormalization()(x + ff)


def build_model(config, lookback, n_price_features, n_macro_features):
    lr = config.get("learning_rate", 0.01)
    loss_fn = config.get("loss", "mse")
    dropout_price = config.get("dropout_price", 0.0)
    dropout_macro = config.get("dropout_macro", 0.0)
    dropout_cross = config.get("dropout_cross", 0.0)
    dropout_fused = config.get("dropout_fused", 0.001)

    price_units = config.get("price_dense_units", 64)
    macro_units = config.get("macro_dense_units", 64)
    fused_units_1 = config.get("fused_dense_units_1", 32)
    fused_units_2 = config.get("fused_dense_units_2", 32)

    attn_heads = config.get("attention_heads", 4)
    attn_key_dim = config.get("attention_key_dim", 16)

    horizon = config.get("horizon", 1)

    price_input = Input(shape=(lookback, n_price_features), name="price_input")
    macro_input = Input(shape=(lookback, n_macro_features), name="macro_input")

    price_enc = _encode(price_input, lookback, price_units, attn_heads, attn_key_dim, dropout_price)
    macro_enc = _encode(macro_input, lookback, macro_units, attn_heads, attn_key_dim, dropout_macro)

    # Price positions attend to the macro sequence.
    cross_attn = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(price_enc, macro_enc)
    cross_attn = Dropout(dropout_cross)(cross_attn)
    cross_attn = LayerNormalization()(cross_attn)

    fused = Concatenate()([price_enc, cross_attn])
    fused = Dense(fused_units_1, activation="relu")(fused)
    fused = Dropout(dropout_fused)(fused)
    fused = Dense(fused_units_2, activation="relu")(fused)

    pooled = GlobalAveragePooling1D()(fused)
    output = Dense(horizon, activation="linear", name="price_prediction")(pooled)

    model = Model(inputs=[price_input, macro_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=["mae"])
    return model
