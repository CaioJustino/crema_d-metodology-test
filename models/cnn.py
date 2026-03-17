import tensorflow as tf

def criarCnn(input_shape, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Bloco 1: Extração de Features de Baixo Nível
        tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        
        # Bloco 2: Features de Médio Nível
        tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # Bloco 3: Features de Alto Nível
        tf.keras.layers.Conv1D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        # Global Average Pooling ajuda muito a reduzir Overfitting em comparação ao Flatten
        tf.keras.layers.GlobalAveragePooling1D(), 

        # Classificador
        tf.keras.layers.Dense(128, activation="relu", 
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model