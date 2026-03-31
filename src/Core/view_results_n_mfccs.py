import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def extrair_metricas_completas():
    res_base_path = "resultados"
    modelos = ["CNN", "CRNN", "RESNET"]
    sementes = [42, 123, 999]
    metricas_alvo = ["accuracy", "recall", "precision", "f1-score"]
    
    pastas = sorted([int(d.split('_')[-1]) for d in os.listdir(res_base_path) if d.startswith("n_mfccs_")])
    
    data = {m: {met: {n: [] for n in pastas} for met in metricas_alvo} for m in modelos}

    for n in pastas:
        for m in modelos:
            for s in sementes:
                path = os.path.join(res_base_path, f"n_mfccs_{n}", f"{m}_history_seed_{s}.pkl")
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        h = pickle.load(f)
                        
                        data[m]["accuracy"][n].append(max(h.get('val_accuracy', [0])))
                        
                        report = h.get('classification_report', {})
                        if isinstance(report, dict) and 'macro avg' in report:
                            data[m]["recall"][n].append(report['macro avg']['recall'])
                            data[m]["precision"][n].append(report['macro avg']['precision'])
                            data[m]["f1-score"][n].append(report['macro avg']['f1-score'])

    return data, pastas, modelos, metricas_alvo

def plotar_benchmark_completo(data, pastas, modelos, metricas):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    cores = {"CNN": "#1f77b4", "CRNN": "#2ca02c", "RESNET": "#d62728"}
    
    for idx, met in enumerate(metricas):
        ax = axes[idx]
        for m in modelos:
            medias = []
            stds = []
            mfccs_validos = []
            
            for n in pastas:
                valores = data[m][met][n]
                if valores:
                    mfccs_validos.append(n)
                    medias.append(np.mean(valores))
                    stds.append(np.std(valores))
            
            if mfccs_validos:
                line = ax.plot(mfccs_validos, medias, label=m, color=cores[m], marker='o', linewidth=2)
                ax.fill_between(mfccs_validos, np.array(medias) - np.array(stds), 
                                np.array(medias) + np.array(stds), color=cores[m], alpha=0.1)

        ax.set_title(f"Média Global: {met.capitalize()}", fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(pastas)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Número de MFCCs")
        ax.set_ylabel("Pontuação (Score)")
        ax.set_ylim(0.3, 0.7)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dados, mfccs, mods, mets = extrair_metricas_completas()
    plotar_benchmark_completo(dados, mfccs, mods, mets)