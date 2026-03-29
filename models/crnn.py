import tensorflow as tf
import numpy as np

def criar_crnn(input_shape, n_classes):
    t, f, c = input_shape
    
    t_after = int(np.ceil(t / 2))
    f_after = int(np.ceil(f / 2))
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        
        tf.keras.layers.Reshape((t_after, f_after * 64)),
        
        tf.keras.layers.LSTM(128, dropout=0.2), 
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model