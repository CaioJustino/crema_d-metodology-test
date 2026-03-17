import numpy as np
import tensorflow as tf
import os
from models.cnn import criarCnn

def train_model(processed_path, n_classes):
    # Carregar dados
    X_train = np.load(os.path.join(processed_path, "X_train.npy"))
    y_train = np.load(os.path.join(processed_path, "y_train.npy"))
    X_val = np.load(os.path.join(processed_path, "X_validation.npy"))
    y_val = np.load(os.path.join(processed_path, "y_validation.npy"))

    # Converter labels para categorical
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val = tf.keras.utils.to_categorical(y_val, n_classes)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = criarCnn(input_shape, n_classes)

    print("Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    model.save("models/emotion_model.h5")
    return model, history