from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dropout, LayerNormalization,
    Concatenate, Dense, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

def build_model(config, lookback, n_price_features, n_macro_features):
    H = config.get("horizon", 1)
    
    base_units = config.get("units", 64)

    # GRU units per layer
    units_price_1 = config.get("units_price_1", base_units)
    units_price_2 = config.get("units_price_2", base_units)
    units_price_3 = config.get("units_price_3", base_units)
    units_macro_1 = config.get("units_macro_1", base_units)
    units_macro_2 = config.get("units_macro_2", base_units)

    # Dense units in fused layers
    fused_dense_1 = config.get("fused_dense_units_1", 128)
    fused_dense_2 = config.get("fused_dense_units_2", 64)

    # Dropouts
    dropout_price_1 = config.get("dropout_price_1", 0.0)
    dropout_price_2 = config.get("dropout_price_2", 0.0)
    dropout_macro = config.get("dropout_macro", 0.001)
    dropout_fused = config.get("dropout_fused", 0.001)

    # Learning rate and loss
    learning_rate = config.get("learning_rate", 1e-4)
    loss_fn = config.get("loss", "huber")

    # Inputs
    price_input = Input(shape=(lookback, n_price_features), name='price_input')
    macro_input = Input(shape=(lookback, n_macro_features), name='macro_input')

    # Price stream
    x_price = LSTM(units_price_1, return_sequences=True)(price_input)
    x_price = Dropout(dropout_price_1)(x_price)
    x_price = LSTM(units_price_2, return_sequences=True)(x_price)
    x_price = Dropout(dropout_price_2)(x_price)
    x_price = LSTM(units_price_3, return_sequences=True)(x_price)
    x_price = LayerNormalization()(x_price)

    # Macro stream
    x_macro = LSTM(units_macro_1, return_sequences=True)(macro_input)
    x_macro = Dropout(dropout_macro)(x_macro)
    x_macro = LSTM(units_macro_2, return_sequences=True)(x_macro)
    x_macro = LayerNormalization()(x_macro)

    # Fusion
    fused = Concatenate()([x_price, x_macro])
    fused = Dense(fused_dense_1, activation='gelu')(fused)
    fused = Dropout(dropout_fused)(fused)
    fused = Dense(fused_dense_2, activation='gelu')(fused)

    pooled = GlobalAveragePooling1D()(fused)
    output = Dense(H, activation='linear', name='price_prediction')(pooled)

    model = Model(inputs=[price_input, macro_input], outputs=output)

    # Loss function
    if loss_fn.lower() == "huber":
        loss = Huber()
    else:
        loss = loss_fn

    model.compile(optimizer=Adam(learning_rate), loss=loss, metrics=['mae', 'mse'])

    return model
