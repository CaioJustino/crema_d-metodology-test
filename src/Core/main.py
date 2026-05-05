import os
import sys
from src.Core.processing import process_and_save
from src.Core.training import train_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def rodar_experimento_mfcc(lista_mfccs: list, qtd_epocas: int, lista_seeds: list):
    RAW_DATA_PATH = "data/raw/AudioWAV"
    BASE_PROCESSED_PATH = "data/processed/AudioMFCC"
    CSV_PATH = "data/processed/processed_results/summaryTable.csv"
    
    modelos = ["CNN", "CRNN", "ResNet"]
    
    for n_mfcc in lista_mfccs:
        print(f"\n{'='*50}")
        print(f"INICIANDO BATERIA: {n_mfcc} MFCCs")
        print(f"{'='*50}")
        
        path_processado = process_and_save(RAW_DATA_PATH, BASE_PROCESSED_PATH, CSV_PATH, n_mfcc)

        for modelo in modelos:
            print(f"\n{'-'*30}\n>>> Arquitetura: {modelo} | MFCCs: {n_mfcc}\n{'-'*30}")
            
            for seed in lista_seeds:
                print(f"--> Executando com Semente Aleatória: {seed}")
                try:
                    train_model(processed_path=path_processado, n_classes=6, arquitetura=modelo, qtdEpocas=qtd_epocas, flexMfccs=n_mfcc, seed=seed)
                except Exception as e:
                    print(f"Erro no treino ({modelo}, {n_mfcc} MFCCs, Seed {seed}): {e}")

if __name__ == "__main__":
    sementes_teste = [10, 42, 55, 101, 123, 202, 303, 314, 404, 505, 606, 666, 707, 777, 808, 909, 999, 1024, 1337, 2024, 2026, 3333, 4444, 5555, 7777, 8888, 9999, 12345, 54321, 98765] 
    mfccs_teste = [2, 4, 6, 8, 10, 13, 15, 20, 30, 40, 62, 82, 102, 128]
    
    rodar_experimento_mfcc(lista_mfccs=mfccs_teste, qtd_epocas=50, lista_seeds=sementes_teste)