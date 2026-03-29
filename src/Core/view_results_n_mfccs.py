import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [seq[-1]] * (max_len - len(seq))
        padded.append(seq)
    return np.array(padded)

def visualizar_resultados():
    res_base_path = "resultados"
    
    n_mfccs = input("Digite o número de MFCCs (ex: 13): ")
    modelo_nome = input("Digite o nome do modelo (ex: CNN, ResNet, CRNN): ").upper()
    
    folder_path = os.path.join(res_base_path, f"n_mfccs_{n_mfccs}")
    search_pattern = os.path.join(folder_path, f"{modelo_nome}_history_seed_*.pkl")
    arquivos = glob.glob(search_pattern)
    
    if not arquivos:
        print(f"Erro: Nenhum arquivo encontrado para o padrão {search_pattern}")
        return

    histories = []
    for arq in arquivos:
        with open(arq, 'rb') as f:
            histories.append(pickle.load(f))

    print(f"Foram encontradas {len(histories)} execuções (seeds) para este modelo.")

    accs, val_accs, losses, val_losses = [], [], [], []
    for h in histories:
        if 'accuracy' in h: accs.append(h['accuracy'])
        if 'val_accuracy' in h: val_accs.append(h['val_accuracy'])
        if 'loss' in h: losses.append(h['loss'])
        if 'val_loss' in h: val_losses.append(h['val_loss'])

    max_epochs = max([len(a) for a in accs])

    accs = pad_sequences(accs, max_epochs)
    val_accs = pad_sequences(val_accs, max_epochs)
    losses = pad_sequences(losses, max_epochs)
    val_losses = pad_sequences(val_losses, max_epochs)

    epocas = np.arange(1, max_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epocas, np.mean(accs, axis=0), label='Treino (Média)', color='blue')
    ax1.fill_between(epocas, np.mean(accs, axis=0) - np.std(accs, axis=0), 
                     np.mean(accs, axis=0) + np.std(accs, axis=0), color='blue', alpha=0.2)
    
    ax1.plot(epocas, np.mean(val_accs, axis=0), label='Validação (Média)', color='orange')
    ax1.fill_between(epocas, np.mean(val_accs, axis=0) - np.std(val_accs, axis=0), 
                     np.mean(val_accs, axis=0) + np.std(val_accs, axis=0), color='orange', alpha=0.2)
    
    ax1.set_title(f'Acurácia com Desvio Padrão - {modelo_nome} ({n_mfccs} MFCCs)')
    ax1.set_xlabel('Época')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epocas, np.mean(losses, axis=0), label='Treino (Média)', color='blue')
    ax2.fill_between(epocas, np.mean(losses, axis=0) - np.std(losses, axis=0), 
                     np.mean(losses, axis=0) + np.std(losses, axis=0), color='blue', alpha=0.2)
    
    ax2.plot(epocas, np.mean(val_losses, axis=0), label='Validação (Média)', color='orange')
    ax2.fill_between(epocas, np.mean(val_losses, axis=0) - np.std(val_losses, axis=0), 
                     np.mean(val_losses, axis=0) + np.std(val_losses, axis=0), color='orange', alpha=0.2)
    
    ax2.set_title(f'Loss com Desvio Padrão - {modelo_nome} ({n_mfccs} MFCCs)')
    ax2.set_xlabel('Época')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualizar_resultados()