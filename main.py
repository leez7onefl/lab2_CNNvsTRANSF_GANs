import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

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
    return pd.DataFrame(data, columns=["Layer (type)", "Output Shape", "Param #",])

# Plotting function
def plot_losses(epoch, d_loss_cnn, g_loss_cnn, d_loss_trans, g_loss_trans):
    st.subheader(f'Epoch {epoch + 1}/{epochs}')
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CNN GAN Loss", "Transformer GAN Loss"))
    
    fig.add_trace(go.Scatter(x=list(range(1, epoch + 2)), y=d_loss_cnn[:epoch + 1], mode='lines+markers', name='D Loss CNN'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, epoch + 2)), y=g_loss_cnn[:epoch + 1], mode='lines+markers', name='G Loss CNN'), row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(1, epoch + 2)), y=d_loss_trans[:epoch + 1], mode='lines+markers', name='D Loss Transformer'), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(1, epoch + 2)), y=g_loss_trans[:epoch + 1], mode='lines+markers', name='G Loss Transformer'), row=1, col=2)

    fig.update_layout(height=500, width=1000, title_text="GAN Training Losses")
    st.plotly_chart(fig)

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

# Initialize Streamlit
st.set_page_config(layout="wide")
st.title("GAN Training Comparison: CNN vs. Transformer")

# Sidebar for parameters
st.sidebar.title("Training Parameters")
latent_dim = st.sidebar.number_input("Latent Dimension", value=100, min_value=1)
data_percentage = st.sidebar.slider("Data Percentage", min_value=0.01, max_value=1.0, value=0.2)
epochs = st.sidebar.slider("Epochs", value=10, min_value=1)
batch_size = st.sidebar.slider("Batch Size", value=16, min_value=1)
alpha_exponent = st.sidebar.slider("Optimizer Alpha Exponent", value=-5, min_value=-10, max_value=-1, step=1)
optimizer_alpha = np.power(10.0, alpha_exponent)
start_training = st.sidebar.button("Start Training")

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
    subset_size = int(x_train.shape[0] * data_percentage)
    x_train_subset = x_train[:subset_size]

    cnn_discriminator.compile(optimizer=tf.keras.optimizers.Adam(optimizer_alpha), loss="binary_crossentropy", metrics=["accuracy"])
    cnn_gan = build_cnn_gan(cnn_generator, cnn_discriminator, latent_dim)

    transformer_discriminator.compile(optimizer=tf.keras.optimizers.Adam(optimizer_alpha), loss="binary_crossentropy", metrics=["accuracy"])
    transformer_gan = build_transformer_gan(transformer_generator, transformer_discriminator, latent_dim)

    d_loss_cnn = []
    g_loss_cnn = []
    d_loss_trans = []
    g_loss_trans = []

    for epoch in range(epochs):
        d_loss_real_cnn, d_loss_fake_cnn, g_loss_cnn_epoch = train_cnn_gan(
            cnn_generator, cnn_discriminator, cnn_gan, x_train_subset, epochs,
            batch_size=batch_size, latent_dim=latent_dim
        )
        d_loss_cnn.append(d_loss_real_cnn + d_loss_fake_cnn)
        g_loss_cnn.append(g_loss_cnn_epoch)

        d_loss_real_trans, d_loss_fake_trans, g_loss_trans_epoch = train_transformer_gan(
            transformer_generator, transformer_discriminator, transformer_gan, x_train_subset,
            epochs, batch_size=batch_size, latent_dim=latent_dim
        )
        d_loss_trans.append(d_loss_real_trans + d_loss_fake_trans)
        g_loss_trans.append(g_loss_trans_epoch)

        plot_losses(epoch, d_loss_cnn, g_loss_cnn, d_loss_trans, g_loss_trans)