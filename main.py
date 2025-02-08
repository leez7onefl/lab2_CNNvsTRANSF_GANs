import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Import models and training functions
from models.cnn_gan.generator import build_cnn_generator
from models.cnn_gan.discriminator import build_cnn_discriminator
from models.transformer_gan.generator import build_transformer_generator
from models.transformer_gan.discriminator import build_transformer_discriminator
from training.train_cnn_gan import build_cnn_gan, train_cnn_gan
from training.train_transformer_gan import build_transformer_gan, train_transformer_gan

# Function to convert model summary to DataFrame
def model_summary_to_dataframe(model):
    data = []
    for layer in model.layers:
        data.append({
            "Layer (type)": layer.name + " (" + layer.__class__.__name__ + ")",
            "Output Shape": str(layer.output.shape),
            "Param #": layer.count_params(),
        })
    return pd.DataFrame(data, columns=["Layer (type)", "Output Shape", "Param #"])

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

# Initialize Streamlit
st.set_page_config(layout="wide")
st.title("GAN Training Comparison: CNN vs. Transformer")

# Sidebar for parameters
st.sidebar.title("Training Parameters")
latent_dim = st.sidebar.slider("Latent Dimension", value=100, min_value=1, max_value=500, step=1)
# data_percentage = st.sidebar.slider("Data Percentage", min_value=0.01, max_value=1.0, value=0.2)
epochs = st.sidebar.slider("Epochs", value=200, min_value=0, max_value=500, step=1)
batch_size = st.sidebar.slider("Batch Size", value=32, min_value=0, max_value=1024, step=32)
alpha_exponent = st.sidebar.slider("Optimizer Alpha Exponent", value=-5, min_value=-10, max_value=-1, step=1)
optimizer_alpha = np.power(10.0, alpha_exponent)
start_training = st.sidebar.button("Start Training")

# Button to generate images
generate_images = st.sidebar.button("Generate Images")

# Build models for summaries
cnn_generator = build_cnn_generator(latent_dim)
cnn_discriminator = build_cnn_discriminator()

transformer_generator = build_transformer_generator(latent_dim)
transformer_discriminator = build_transformer_discriminator()

# Display model summaries as DataFrame
st.subheader("CNN Generator Model Summary")
st.dataframe(model_summary_to_dataframe(cnn_generator))

st.subheader("CNN Discriminator Model Summary")
st.dataframe(model_summary_to_dataframe(cnn_discriminator))

st.subheader("Transformer Generator Model Summary")
st.dataframe(model_summary_to_dataframe(transformer_generator))

st.subheader("Transformer Discriminator Model Summary")
st.dataframe(model_summary_to_dataframe(transformer_discriminator))

if start_training:
    # Compile models
    cnn_discriminator.compile(optimizer=tf.keras.optimizers.Adam(optimizer_alpha), loss="binary_crossentropy", metrics=["accuracy"])
    cnn_gan = build_cnn_gan(cnn_generator, cnn_discriminator, latent_dim)

    transformer_discriminator.compile(optimizer=tf.keras.optimizers.Adam(optimizer_alpha), loss="binary_crossentropy", metrics=["accuracy"])
    transformer_gan = build_transformer_gan(transformer_generator, transformer_discriminator, latent_dim)

    # Initialize lists to store loss values
    d_loss_cnn = []
    g_loss_cnn = []
    d_loss_trans = []
    g_loss_trans = []

    # Training CNN GAN
    trained_cnn_generator, d_loss_real_cnn, d_loss_fake_cnn, g_loss_cnn_epoch = train_cnn_gan(
        cnn_generator, 
        cnn_discriminator, 
        cnn_gan, x_train, 
        epochs,
        batch_size=batch_size, 
        latent_dim=latent_dim
    )
    d_loss_cnn.append(d_loss_real_cnn + d_loss_fake_cnn)
    g_loss_cnn.append(g_loss_cnn_epoch)

    # Training Transformer GAN
    trained_transformer_generator, d_loss_real_trans, d_loss_fake_trans, g_loss_trans_epoch = train_transformer_gan(
        transformer_generator, 
        transformer_discriminator, 
        transformer_gan, 
        x_train,
        epochs, 
        batch_size=batch_size, 
        latent_dim=latent_dim
    )
    d_loss_trans.append(d_loss_real_trans + d_loss_fake_trans)
    g_loss_trans.append(g_loss_trans_epoch)

    # Save trained models 
    trained_cnn_generator.save("cnn_generator.keras")
    trained_transformer_generator.save("transformer_generator.keras")

    # Indicate successful completion
    st.success("Models saved successfully!")

if generate_images:
    st.subheader("Generated Images")

    # Load saved models
    cnn_generator = tf.keras.models.load_model("cnn_generator.keras")
    transformer_generator = tf.keras.models.load_model("transformer_generator.keras")

    # Generate random latent vectors
    random_latent_vectors = np.random.normal(size=(1, latent_dim))

    # Generate images using the CNN generator
    cnn_generated_images = cnn_generator(random_latent_vectors)
    cnn_generated_images = (cnn_generated_images * 255).numpy().astype(np.uint8)

    st.write("Images generated by the CNN GAN:")

    # Display CNN generated images
    for i, img in enumerate(cnn_generated_images):
        st.image(img, caption=f'CNN Image {i+1}', use_container_width=True)

    # Generate images using the Transformer generator
    transformer_generated_images = transformer_generator(random_latent_vectors)
    transformer_generated_images = (transformer_generated_images * 255).numpy().astype(np.uint8)

    st.write("Images generated by the Transformer GAN:")

    # Display Transformer generated images
    for i, img in enumerate(transformer_generated_images):
        st.image(img, caption=f'Transformer Image {i+1}', use_container_width=True)