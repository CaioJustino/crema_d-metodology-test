import os
import time
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from models.cnn import criar_cnn
from models.resnet import criar_resnet
from models.crnn import criar_crnn

def log_exec_time_pkl(arquitetura, n_mfccs, seed, elapsed_time):
    pkl_path = os.path.join("resultados", "tempos_execucao.pkl")
    os.makedirs("resultados", exist_ok=True)
    dados_tempo = {}
    
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                dados_tempo = pickle.load(f)
        except (EOFError, FileNotFoundError): pass

    if arquitetura not in dados_tempo: dados_tempo[arquitetura] = {}
    if n_mfccs not in dados_tempo[arquitetura]: dados_tempo[arquitetura][n_mfccs] = {}
    
    dados_tempo[arquitetura][n_mfccs][seed] = float(f"{elapsed_time:.2f}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(dados_tempo, f)

def train_model(processed_path, n_classes, arquitetura, qtdEpocas, flexMfccs, seed):
    tf.keras.utils.set_random_seed(seed)
    
    X_train = np.load(os.path.join(processed_path, "X_train.npy"))
    y_train = np.load(os.path.join(processed_path, "y_train.npy"))
    X_val = np.load(os.path.join(processed_path, "X_validation.npy"))
    y_val = np.load(os.path.join(processed_path, "y_validation.npy"))

    # Escalonamento
    b, t, f = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, f)).reshape(-1, t, f)
    X_val = scaler.transform(X_val.reshape(-1, f)).reshape(-1, t, f)

    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, n_classes)

    X_train_in = X_train[..., np.newaxis]
    X_val_in = X_val[..., np.newaxis]
    input_shape = (t, f, 1)

    arch = arquitetura.upper()
    if arch == 'CNN': model = criar_cnn(input_shape, n_classes)
    elif arch == 'RESNET': model = criar_resnet(input_shape, n_classes)
    elif arch == 'CRNN': model = criar_crnn(input_shape, n_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), 
                  loss="categorical_crossentropy", metrics=["accuracy"])

    start_time = time.time()
    history = model.fit(
        X_train_in, y_train_cat, 
        validation_data=(X_val_in, y_val_cat), 
        epochs=qtdEpocas, batch_size=32, 
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)], 
        verbose=1
    )
    elapsed_time = time.time() - start_time
    
    y_pred = np.argmax(model.predict(X_val_in), axis=1)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    macro_metrics = report.get('macro avg', {})
    
    idx_best = np.argmax(history.history['val_accuracy'])
    metrics_to_save = {
        'train_acc': history.history['accuracy'][idx_best],
        'val_acc': history.history['val_accuracy'][idx_best],
        'val_precision': macro_metrics.get('precision', 0),
        'val_recall': macro_metrics.get('recall', 0),
        'val_f1': macro_metrics.get('f1-score', 0),
        'classification_report': report
    }

    log_exec_time_pkl(arch, flexMfccs, seed, elapsed_time)
    
    res_path = os.path.join("resultados", f"n_mfccs_{flexMfccs}", f"seed_{seed}")
    os.makedirs(res_path, exist_ok=True)
    with open(os.path.join(res_path, f"{arch}_history.pkl"), 'wb') as f:
        pickle.dump(metrics_to_save, f)