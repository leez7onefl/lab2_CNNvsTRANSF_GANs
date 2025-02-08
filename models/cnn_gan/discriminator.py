from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU
import tensorflow as tf

def build_cnn_discriminator(input_shape=(28, 28, 1), alpha=0.2):

    model = tf.keras.Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=input_shape),
        LeakyReLU(alpha=alpha),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=alpha),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])
    
    return model