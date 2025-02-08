import tensorflow as tf
import numpy as np

def build_cnn_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss='binary_crossentropy')
    return gan

def train_cnn_gan(generator, discriminator, gan, x_train, epochs, batch_size, latent_dim):
    d_loss_real_accum = 0
    d_loss_fake_accum = 0
    g_loss_accum = 0

    batch_count = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):  # Change this to match the batch_count
            # Generate fake images
            noise = tf.random.normal([batch_size, latent_dim])
            fake_images = generator.predict(noise)

            # Get real images
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            
            # Create labels
            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            
            d_loss_real_accum += d_loss_real[0]
            d_loss_fake_accum += d_loss_fake[0]

            # Train generator
            misleading_labels = tf.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, misleading_labels)
            
            g_loss_accum += g_loss

        # Print losses
        d_loss = d_loss_real_accum / batch_count + d_loss_fake_accum / batch_count
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss_accum / batch_count}")

    return d_loss_real_accum / batch_count, d_loss_fake_accum / batch_count, g_loss_accum / batch_count