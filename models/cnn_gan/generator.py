from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
import tensorflow as tf

def build_cnn_generator(latent_dim, img_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    model.add(Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(Conv2DTranspose(img_shape[-1], kernel_size=7, activation="tanh", padding="same"))
    return model