# Documentation Exercice Transformers

### Partie 1: Création et Gestion du Vocabulaire

```python
vocab = ['I', 'love', 'computer', 'vision', 'even', 'if', 'i', 'loose', 'my', 'sanity', 'over', 'it']
random.shuffle(vocab)
```

- **Explication**: `vocab` constitue une liste de mots formant le vocabulaire utilisé pour créer des séquences. La fonction `random.shuffle` effectue un mélange aléatoire des mots.

- **Concept**: Le mélange n'a pas d'impact ici, mais je voulais comparer les résultats avec, j'ai compris qu'il peut s'avérer utile pour prévenir les biais dans d'autres contextes mais pas ici.

```python
vocab_size = len(vocab)
```

- **Explication**: `vocab_size` représente le nombre total de mots dans le vocabulaire.

- **Concept**: La taille est essentielle pour définir les dimensions des matrices d'embeddings et des couches de réseau, pour que chaque mot soit représenté correctement dans le modèle.
---
### Partie 2: Conversion de la Séquence en Identifiants

```python
sequence = ['I', 'love', 'computer', 'vision', 'even', 'if', 'i', 'loose', 'my', 'sanity', 'over', 'it']
input_sequence_ids = [word_to_id[word] for word in sequence]
input_sequence_ids = tf.constant([input_sequence_ids])
```

- **Explication**: La liste `sequence` est transformée en identifiants numériques à l'aide de `word_to_id`. La conversion en tenseur avec `tf.constant` prépare la séquence pour son utilisation dans TensorFlow.

- **Concept**: Un tenseur formalise mathématiquement la liste des identifiants pour les opérations. Les tenseurs sont similaire à des matrices à plusieurs dimensions. Un tenseur en dim. 1 est un vecteur, en dim. 2 une matrice ligne*colonne, en dim. 3 ligne\*colonne\*profondeur, etc
---
### Partie 3: Paramètres des Transformers

```python
embed_dim = 4  # Embedding dimension
num_heads = 1  # Number of attention heads
```

- **Explication**: `embed_dim` spécifie la dimensionalité de l'espace d'embedding des mots. `num_heads` dénombre les têtes dans l'attention multi-têtes, ici réduites à une pour simplification.

- **Concept**: L'embedding place des mots similaires à proximité dans un espace multi-dimensionnel. Il faut voir les bibliothèque d'embedding comme des outils qui convertissent les mots en vecteurs qui représentent bien leur sens. (Ex. classique du prince+femme=princesse)
---
### Partie 4: Création des Embeddings

```python
embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)
```

- **Explication**: Une couche d'embedding convertit les identifiants numériques en vecteurs dans un espace de dimension `embed_dim`.

- **Concept**: Chaque mot devient un vecteur différent en fonction de son sens dans l'espace défini par `embed_dim`.

```python
embedded_input = embedding_layer(input_sequence_ids)
```

- **Explication**: `embedded_input` est le résultat des mots de la séquence passés dans la couche d'embedding.
---
### Partie 5: Calcul de l'Attention Simple

```python
    query_dense = tf.keras.layers.Dense(embed_dim)
    key_dense = tf.keras.layers.Dense(embed_dim)
    value_dense = tf.keras.layers.Dense(embed_dim)
```

- **Explication**: Des couches `Dense` transforment les entrées en formats `Q` (Query), `K` (Key), et `V` (Value).

- **Concept**: Les transformations query `Q`, key `K`, et value `V` aident à déterminer quelles parties de la séquence nécessitent une attention particulière. 

```python
    Q = query_dense(inputs)
    K = key_dense(inputs)
    V = value_dense(inputs)
```

- **Explication**: Transformation des entrées par les couches `query_dense`, `key_dense`, et `value_dense` pour produire `Q`, `K`, et `V`.

```python
    attention_scores = tf.matmul(Q, K, transpose_b=True)
    attention_scores /= tf.math.sqrt(tf.cast(embed_dim, tf.float32))
```

- **Explication**: Les scores d'attention sont calculés par multiplication de `Q` avec `K` transposé. Ils sont ensuite normalisés par la racine carrée de `embed_dim` pour stabilité numérique.

- **Concept**: Cette multiplication évalue comment chaque mot de la séquence se connecte avec d'autres, et la division normalise les valeurs.

```python
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
```

- **Explication**: La fonction `softmax` convertit les scores en probabilités, chaque unité s'additionnant à 1 pour chaque mot.

- **Concept**: L'application de softmax transforme les scores en probabilités comparables aux prévisions dans une compétition.

```python
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
```

- **Explication**: Le produit des poids d'attention avec `V` génère l'output final, accentuant les informations pertinentes, avec un retour des poids d'attention.
---
### Partie 6: Exécution de l'Attention et Affichage

```python
attention_output, attention_weights = compute_self_attention(embedded_input, embed_dim)
```

- **Explication**: La fonction `compute_self_attention` est appliquée aux inputs embed, produisant le résultat et les poids d'attention.

```python
def plot_attention(attention_weights, sequence):
    attention_weights = attention_weights.numpy()[0]
    plt.figure(figsize=(6, 6))
    sns.heatmap(attention_weights, xticklabels=sequence, yticklabels=sequence, cmap='viridis', annot=True)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.show()
```

- **Explication**: Cette fonction génère une représentation visuelle des poids d'attention sous forme de "heatmap", illustrant l'importance accordée à chaque mot.
