import tensorflow as tf

def residual_block_2d(x, filters, downsample=False):
    shortcut = x
    strides = (2, 2) if downsample else (1, 1)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if downsample or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def criar_resnet(input_shape, n_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = residual_block_2d(x, 64)
    x = residual_block_2d(x, 128, downsample=True)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)