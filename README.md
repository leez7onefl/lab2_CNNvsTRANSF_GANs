# Rapport : Réseaux Antagonistes Génératifs (GANs), Base CNN vs base Transformer

## Contexte du Projet

Ce rapport détaille l'implémentation et l'évaluation de GANs basés sur des architectures CNN et Transformers, dans le cadre du cours "Generative AI for Computer Vision", niveau ING3 (Ingénieur 3).

---

## Objectifs

1. Implémentez un GAN basé sur CNN pour la génération d'images réalistes.
2. Étudiez l'utilisation des architectures basées sur Transformers dans les GANs pour la modélisation générative.
3. Comparez les performances et les résultats visuels des différentes architectures de GAN.
4. Réfléchissez aux forces et aux défis de l'utilisation des CNN par rapport aux Transformers dans les GANs.

---

## 1. GANs Basés sur CNN

### Code du Discriminateur

```python
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
```

**Explication du Code :**

- **Conv2D** : Applique une transformation convolutive, pour extraire les caractéristiques d'image.
- **LeakyReLU** : Activation qui permet un flux de gradient dans des régions de non-activité, évitant le *gradient vanishing*.
- **Flatten** : Transforme la matrice d'image en un vecteur, élément nécessaire avant l'application de la couche dense.
- **Dense** : Effectue la classification binaire (réel vs faux) avec l'activation sigmoid pour donner une probabilité.

### Code du Générateur

```python
def build_cnn_generator(latent_dim, img_shape=(28, 28, 1)): 
    model = tf.keras.Sequential() 
    model.add(Dense(7 * 7 * 256, input_dim=latent_dim)) 
    model.add(Reshape((7, 7, 256))) 
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu")) 
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")) 
    model.add(Conv2DTranspose(img_shape[-1], kernel_size=7, activation="tanh", padding="same")) 
     
    return model
```

**Explication du Code :**

- **Dense** : Augmente la dimensionnalité du bruit latent pour initialiser la génération de caractéristiques.
- **Reshape** : Ajuste la sortie de la couche dense en matrice 3D adaptée à la convolution.
- **Conv2DTranspose** : Réalise l'échantillonnage inverse pour transformer la matrice 3D en une image.
- **Activation tanh** : Normalise les pixels entre -1 et 1, standard pour les images générées.

---

## 2. GANs Basés sur Transformers

### Code du Discriminateur

```python
def build_transformer_discriminator(input_shape=(28, 28, 1), num_heads=4, key_dim=128): 
    inputs = Input(shape=input_shape) 
    x = Flatten()(inputs) 
    x = Dense(49 * 128, activation="relu")(x)
    x = Reshape((49, 128))(x)
 
    position_encoding = tf.range(start=0, limit=49, delta=1) 
    position_embedding = Embedding(input_dim=49, output_dim=128)(position_encoding) 
    x += position_embedding
 
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x) 
    x = LayerNormalization()(x)
 
    x = Flatten()(x) 
    x = Dense(128, activation="relu")(x) 
    x = Dense(1, activation="sigmoid")(x)
 
    model = Model(inputs, x, name="Transformer_Discriminator") 
    return model
```

**Explication du Code :**

- **MultiHeadAttention** : Capture les relations multi-échelles entre les caractéristiques.
- **LayerNormalization** : Stabilise les transformations de chaque calque, améliorant la convergence.
- **Positional Embedding** : Intègre l'information de la position

### Code du Générateur

```python
def build_transformer_generator(latent_dim=100, num_heads=4, key_dim=128, emb_dim=128): 
    inputs = Input(shape=(latent_dim,)) 
    x = Dense(7 * 7 * emb_dim, activation="relu")(inputs) 
    x = Reshape((49, emb_dim))(x)
 
    position_encoding = tf.range(start=0, limit=49, delta=1) 
    position_embedding = tf.keras.layers.Embedding(input_dim=49, output_dim=emb_dim)(position_encoding) 
    x += position_embedding
 
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x) 
    x = LayerNormalization()(x)
 
    x = Dense(emb_dim, activation="relu")(x) 
    x = LayerNormalization()(x)
 
    x = Reshape((7, 7, emb_dim))(x) 
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x) 
    x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)
 
    model = Model(inputs, x, name="Transformer_Generator") 
    return model
```

**Explication du Code :**

- **Project Noise** : Étend le bruit latent à des caractéristiques significatives.
- **MultiHeadAttention** : Favorise le mélange spatial pour synthétiser des images riches en détails.
- **Conv2DTranspose** : Convertit l'espace de caractéristiques en images.

---

## 3. Entrainement des GANs

### Entrainement CNN GAN

```python
def train_cnn_gan(generator, discriminator, gan, x_train, epochs, batch_size, latent_dim): 
    d_loss_real_accum = 0 
    d_loss_fake_accum = 0 
    g_loss_accum = 0 
 
    batch_count = x_train.shape[0] // batch_size 
 
    for epoch in range(epochs): 
        for _ in range(batch_count): 
            noise = tf.random.normal([batch_size, latent_dim]) 
            fake_images = generator.predict(noise) 
 
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)] 
             
            real_labels = tf.ones((batch_size, 1)) 
            fake_labels = tf.zeros((batch_size, 1)) 
 
            d_loss_real = discriminator.train_on_batch(real_images, real_labels) 
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels) 
             
            d_loss_real_accum += d_loss_real[0] 
            d_loss_fake_accum += d_loss_fake[0] 
 
            misleading_labels = tf.ones((batch_size, 1)) 
            g_loss = gan.train_on_batch(noise, misleading_labels) 
             
            g_loss_accum += g_loss 
 
        d_loss = d_loss_real_accum / batch_count + d_loss_fake_accum / batch_count 
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss_accum / batch_count}") 
 
    return d_loss_real_accum / batch_count, d_loss_fake_accum / batch_count, g_loss_accum / batch_count
```

### Entrainement Transformer GAN

```python
def train_transformer_gan(generator, discriminator, gan, x_train, epochs, batch_size, latent_dim): 
    for epoch in range(epochs): 
        d_loss_real_accum = 0 
        d_loss_fake_accum = 0 
        g_loss_accum = 0 
 
        batch_count = x_train.shape[0] // batch_size 
 
        for _ in range(batch_count): 
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim)) 
            generated_images = generator.predict(noise) 
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)] 
 
            labels_real = np.ones((batch_size, 1)) 
            labels_fake = np.zeros((batch_size, 1)) 
 
            d_loss_real = discriminator.train_on_batch(image_batch, labels_real) 
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake) 
 
            d_loss_real_accum += d_loss_real[0] 
            d_loss_fake_accum += d_loss_fake[0] 
 
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim)) 
            labels_gan = np.ones((batch_size, 1)) 
            g_loss = gan.train_on_batch(noise, labels_gan) 
 
            g_loss_accum += g_loss 
 
        avg_d_loss_real = d_loss_real_accum / batch_count 
        avg_d_loss_fake = d_loss_fake_accum / batch_count 
        avg_g_loss = g_loss_accum / batch_count 
 
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {avg_d_loss_real + avg_d_loss_fake}, G Loss: {avg_g_loss}") 
 
    return avg_d_loss_real, avg_d_loss_fake, avg_g_loss
```


**Explication des Étapes d'Entraînement :**

- La génération de bruit est utilisée pour simuler des images factices.
- Les discriminants sont formés sur des lots d'images réelles et fausses pour minimiser les pertes.
- L'entraînement fait varier les poids du générateur pour concevoir de meilleures fausses images capables de tromper le discriminateur.
---

## 4. Capture d'écran des résultats

**Résultats de l'entraînement**

![models_summaries](https://github.com/user-attachments/assets/efb44301-3759-4f90-b891-6e1c54c2d8e7)

*Merci de noter qu'ici les epochs sont à multiplier par 10 - confusion dans l'affichage*

![epoch0](https://github.com/user-attachments/assets/6e20b465-049a-4e1d-b91f-2dbb57e64a4f)

![epoch1](https://github.com/user-attachments/assets/83ccffe1-60b7-49f2-a3d1-4d77fe5a0a4b)

![epoch2](https://github.com/user-attachments/assets/d584dcdb-4b55-4253-aa07-bd1e0b38bb8e)

![epoch3](https://github.com/user-attachments/assets/a3e8e47a-4d82-4ba8-8e81-5c5a9652319b)

![epoch4](https://github.com/user-attachments/assets/498e3df9-b9ca-4b16-a5e1-faf7ce5a1f61)

- *Apparition d'une forme convergente*

![epoch5](https://github.com/user-attachments/assets/14ea11cd-aacd-4ac7-8bbc-b0b421f541d1)

![epoch6](https://github.com/user-attachments/assets/971b7357-98ef-49b1-bcf3-a15f87119011)

![epoch7](https://github.com/user-attachments/assets/e4c5f70a-b7f5-49d0-8688-3c8cace768ba)

![epoch8](https://github.com/user-attachments/assets/ec1732df-ce64-4a26-84c9-14dcd8973053)

![epoch9](https://github.com/user-attachments/assets/776e8471-34c8-4229-9390-72895e0718a3)

- *Générateur et discriminateur sont assez stables --> résultats corrects*

**Résultats du générateur**

![image](https://github.com/user-attachments/assets/43f1eccd-97b6-4926-908d-80151bb4973d)

*Les résultats démontrent un problème*

*update1:modifications sur le codes faîtes, le problème ne vien pas des hyperparamètres*

*update2:Le problème ne vient pas des données ou de la façon dont elles sontu tilisées*

---
## Conclusions

Les architectures CNN se sont avérées performantes pour capturer les détails locaux, tandis que les Transformers ont facilité la modélisation des relations entre les pixels à grande échelle. Les deux modèles présentent des forces et des limites distinctes dans le contexte de la génération d'images, offrant des approches complémentaires pour les tâches basées sur des données visuelles.

### Informations Personnelles

**Léonard Gonzalez**  
Étudiant en ING3 Big Data & Machine Learning  
EFREI - Paris Panthéon-Assas University
