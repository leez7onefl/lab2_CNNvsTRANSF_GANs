import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, MultiHeadAttention, LayerNormalization, Conv2DTranspose
from tensorflow.keras.models import Model

def build_transformer_generator(latent_dim=100, num_heads=4, key_dim=128, emb_dim=128):
 
    inputs = Input(shape=(latent_dim,))
    
    # Project noise into a higher-dimensional space
    x = Dense(7 * 7 * emb_dim, activation="relu")(inputs)
    x = Reshape((49, emb_dim))(x)  # Reshape to (7x7 patches, emb_dim features)

    # Positional Encoding
    position_encoding = tf.range(start=0, limit=49, delta=1)
    position_embedding = tf.keras.layers.Embedding(input_dim=49, output_dim=emb_dim)(position_encoding)
    x += position_embedding

    # Multi-Head Self-Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization()(x)

    # Feedforward network
    x = Dense(emb_dim, activation="relu")(x)
    x = LayerNormalization()(x)

    # Reshape and upsample to image dimensions
    x = Reshape((7, 7, emb_dim))(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    model = Model(inputs, x, name="Transformer_Generator")
    return model