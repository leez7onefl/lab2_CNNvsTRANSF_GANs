import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Embedding, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model

def build_transformer_discriminator(input_shape=(28, 28, 1), num_heads=4, key_dim=128):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(49 * 128, activation="relu")(x)  # Adjust dense layer output
    x = Reshape((49, 128))(x)  # Reshape correctly

    # Positional Encoding
    position_encoding = tf.range(start=0, limit=49, delta=1)
    position_embedding = Embedding(input_dim=49, output_dim=128)(position_encoding)
    x += position_embedding

    # Multi-Head Self-Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization()(x)

    # Classification layer
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, x, name="Transformer_Discriminator")
    return model