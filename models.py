"""Model architectures (paper Sec. III-D and Table I).

Main model (BiLSTM + GloVe):
  1. Embedding layer initialised with GloVe vectors
  2. Bidirectional LSTM (return_sequences=True)
  3. GlobalMaxPooling1D
  4. Dense + ReLU
  5. Dropout
  6. Dense(1) + sigmoid

The BiGRU, GRU and LSTM baselines from Table I swap only layer 2.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

import config

RNN_VARIANTS = ("bilstm", "bigru", "gru", "lstm")
DISPLAY_NAMES = {
    "bilstm": "BiLSTM + GloVe",
    "bigru": "BiGRU + GloVe",
    "gru": "GRU + GloVe",
    "lstm": "LSTM + GloVe",
}


def _recurrent_layer(variant: str):
    units = config.RNN_UNITS
    if variant == "bilstm":
        return layers.Bidirectional(layers.LSTM(units, return_sequences=True), name="bilstm")
    if variant == "bigru":
        return layers.Bidirectional(layers.GRU(units, return_sequences=True), name="bigru")
    if variant == "lstm":
        return layers.LSTM(units, return_sequences=True, name="lstm")
    if variant == "gru":
        return layers.GRU(units, return_sequences=True, name="gru")
    raise ValueError(f"Unknown variant {variant!r}; choose from {RNN_VARIANTS}")


def build_model(variant: str, embedding_matrix) -> tf.keras.Model:
    vocab_size, dim = embedding_matrix.shape
    model = models.Sequential(
        [
            layers.Input(shape=(config.MAX_LEN,), name="tokens"),
            layers.Embedding(
                vocab_size,
                dim,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=config.TRAINABLE_EMBEDDINGS,
                name="glove_embedding",
            ),                                                       # Layer 1
            _recurrent_layer(variant),                               # Layer 2
            layers.GlobalMaxPooling1D(name="global_max_pool"),       # Layer 3
            layers.Dense(config.DENSE_UNITS, activation="relu", name="dense_relu"),  # Layer 4
            layers.Dropout(config.DROPOUT, name="dropout"),          # Layer 5
            layers.Dense(1, activation="sigmoid", name="output"),    # Layer 6
        ],
        name=f"{variant}_glove",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=config.LOSS,
        metrics=["accuracy", metrics.Precision(name="precision"), metrics.Recall(name="recall")],
    )
    return model
