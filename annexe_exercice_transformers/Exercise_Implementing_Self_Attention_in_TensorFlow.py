import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Define a small vocabulary and sequence
vocab = ['I', 'love', 'computer', 'vision', 'even', 'if', 'i', 'loose', 'my', 'sanity', 'over', 'it']
random.shuffle(vocab)
vocab_size = len(vocab)
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

# Define a sample input sequence
sequence = ['I', 'love', 'computer', 'vision', 'even', 'if', 'i', 'loose', 'my', 'sanity', 'over', 'it']
input_sequence_ids = [word_to_id[word] for word in sequence]
input_sequence_ids = tf.constant([input_sequence_ids])

# Hyperparameters
embed_dim = 4  # Embedding dimension
num_heads = 1  # Number of attention heads

# Initializing embedding layer
embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)

# Obtain embeddings
embedded_input = embedding_layer(input_sequence_ids)

def compute_self_attention(inputs, embed_dim):
    # Define the dense layers
    query_dense = tf.keras.layers.Dense(embed_dim)
    key_dense = tf.keras.layers.Dense(embed_dim)
    value_dense = tf.keras.layers.Dense(embed_dim)

    # Linear transformations
    Q = query_dense(inputs)
    K = key_dense(inputs)
    V = value_dense(inputs)

    # Compute attention scores
    attention_scores = tf.matmul(Q, K, transpose_b=True)
    attention_scores /= tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    # Apply Softmax to get weights
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)

    # Weighted sum of values
    output = tf.matmul(attention_weights, V)
    return output, attention_weights

# Compute self-attention
attention_output, attention_weights = compute_self_attention(embedded_input, embed_dim)

# Visualization of attention weights
def plot_attention(attention_weights, sequence):
    attention_weights = attention_weights.numpy()[0]
    plt.figure(figsize=(6, 6))
    sns.heatmap(attention_weights, xticklabels=sequence, yticklabels=sequence, cmap='viridis', annot=True)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.show()

plot_attention(attention_weights, sequence)