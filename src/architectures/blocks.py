"""Building blocks shared by the Keras models."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import AdamW


class PositionalEncoding(Layer):
    """Fixed sinusoidal position information (Vaswani et al., 2017).

    Self-attention does not know the order of its inputs; without this the
    model cannot tell the most recent day from the oldest one in the window.
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


class LastStep(Layer):
    """Representation of the most recent day in the window.

    The forecast is for the day after the window, so the last position is
    the natural summary; averaging over all 30 days dilutes it.
    """

    def call(self, x):
        return x[:, -1, :]


def embed(x, d_model: int, length: int):
    """Linear projection to ``d_model`` plus positional encoding."""
    return PositionalEncoding(length, d_model)(Dense(d_model)(x))


def encoder_block(x, heads: int, key_dim: int, ff_dim: int, dropout: float):
    """Self-attention and feed-forward sublayers, each with residual and layer norm."""
    attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = LayerNormalization()(x + Dropout(dropout)(attn))
    ff = Dense(x.shape[-1])(Dropout(dropout)(Dense(ff_dim, activation="gelu")(x)))
    return LayerNormalization()(x + Dropout(dropout)(ff))


def cross_block(query, memory, heads: int, key_dim: int, ff_dim: int, dropout: float):
    """Queries from one stream attend to another stream; residual, norm, feed-forward."""
    attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout)(query, memory)
    x = LayerNormalization()(query + Dropout(dropout)(attn))
    ff = Dense(x.shape[-1])(Dropout(dropout)(Dense(ff_dim, activation="gelu")(x)))
    return LayerNormalization()(x + Dropout(dropout)(ff))


def head(x, units: int, dropout: float, horizon: int = 1):
    """Two dense layers and a linear output."""
    x = Dense(units, activation="gelu")(x)
    x = Dropout(dropout)(x)
    x = Dense(max(units // 2, 8), activation="gelu")(x)
    return Dense(horizon, activation="linear", name="return_forecast")(x)


def compile_model(model, config: dict):
    """AdamW with gradient clipping; Huber loss by default (daily returns have fat tails)."""
    loss = Huber(delta=config.get("huber_delta", 1.0)) if config.get("loss", "huber").lower() == "huber" else config["loss"]
    model.compile(
        optimizer=AdamW(
            learning_rate=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
            clipnorm=config.get("clipnorm", 1.0),
        ),
        loss=loss,
        metrics=["mae"],
    )
    return model
