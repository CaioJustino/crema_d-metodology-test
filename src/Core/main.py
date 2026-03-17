import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.Core.processing import process_and_save
from src.Core.training import train_model

def main():
    RAW_DATA_PATH = "data/raw/AudioWAV"
    PROCESSED_DATA_PATH = "data/processed/AudioMFCC"
    CSV_PATH = "data/processed/processed_results/summaryTable.csv"
    
    # Se der erro no processamento anterior, delete a pasta AudioMFCC manualmente 
    # ou use esta verificação:
    if not os.path.exists(os.path.join(PROCESSED_DATA_PATH, "X_train.npy")):
        print("Extraindo MFCCs seguindo o summaryTable.csv...")
        process_and_save(RAW_DATA_PATH, PROCESSED_DATA_PATH, CSV_PATH)
    else:
        print("Dados processados encontrados. Pulando extração.")

    # Treinamento
    n_classes = 6 
    train_model(PROCESSED_DATA_PATH, n_classes)
    
    print("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()