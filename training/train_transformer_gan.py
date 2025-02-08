import tensorflow as tf
import numpy as np

def build_transformer_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss='binary_crossentropy')
    return gan

def train_transformer_gan(generator, discriminator, gan, x_train, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        d_loss_real_accum = 0
        d_loss_fake_accum = 0
        g_loss_accum = 0

        batch_count = x_train.shape[0] // batch_size

        for _ in range(batch_count):
            # Train Discriminator
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            generated_images = generator.predict(noise)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(image_batch, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

            d_loss_real_accum += d_loss_real[0]
            d_loss_fake_accum += d_loss_fake[0]

            # Train Generator
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            labels_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, labels_gan)

            g_loss_accum += g_loss

        # Average the accumulated losses over the batches
        avg_d_loss_real = d_loss_real_accum / batch_count
        avg_d_loss_fake = d_loss_fake_accum / batch_count
        avg_g_loss = g_loss_accum / batch_count

        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {avg_d_loss_real + avg_d_loss_fake}, G Loss: {avg_g_loss}")

    return avg_d_loss_real, avg_d_loss_fake, avg_g_loss