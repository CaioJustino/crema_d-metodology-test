import os
import numpy as np
import librosa
import collections

SR = 16000
DURATION = 3.0
LABELS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

def get_splits_from_dir(raw_path):
    """Lê os arquivos diretamente do diretório sem depender de CSV."""
    bad_files = {'1040_ITH_SAD_XX', '1006_TIE_NEU_XX', '1013_WSI_DIS_XX', '1017_IWW_FEA_XX'}
    valid_items = []
    
    if not os.path.exists(raw_path):
        print(f"Erro: O diretório '{raw_path}' não existe. Crie a pasta e adicione os áudios .wav nela.")
        return {}

    for filename in os.listdir(raw_path):
        if filename.endswith(".wav"):
            wav_name = filename.replace('.wav', '')
            if wav_name not in bad_files:
                speaker_id = wav_name.split('_')[0]
                valid_items.append((wav_name, speaker_id))

    if not valid_items:
        print(f"Erro: O diretório '{raw_path}' existe, mas está vazio ou sem arquivos .wav válidos.")
        return {}

    groups = sorted(set(g for _, g in valid_items))
    np.random.RandomState(42).shuffle(groups)
    
    train_end, val_end = int(0.7 * len(groups)), int(0.8 * len(groups))
    group_to_split = {g: ('train' if i < train_end else 'validation' if i < val_end else 'test') 
                      for i, g in enumerate(groups)}

    splits = collections.defaultdict(list)
    for wav, gid in valid_items:
        if gid in group_to_split: splits[group_to_split[gid]].append(wav)
        
    return splits

def extract_mfcc(file_path, n_mfcc):
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        audio = librosa.util.fix_length(audio, size=int(SR * DURATION))
        n_mels = max(128, n_mfcc)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=n_mels)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)
        return mfcc.T 
    except Exception as e:
        print(f"Erro ao extrair de {file_path}: {e}")
        return None

def process_and_save(raw_path, base_processed_path, n_mfcc):
    target_folder = f"{base_processed_path}_{n_mfcc}"
    
    # Se a pasta já existir e tiver arquivos, assume que já foi processada
    if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
        return target_folder

    print(f"Extraindo {n_mfcc} MFCCs a partir do diretório '{raw_path}'...")
    
    splits = get_splits_from_dir(raw_path)
    if not splits:
        return None

    os.makedirs(target_folder, exist_ok=True)
    total_sucessos = 0

    for name, files in splits.items():
        X, y = [], []
        for wav in files:
            path = os.path.join(raw_path, f"{wav}.wav")
            parts = wav.split('_')
            feat = extract_mfcc(path, n_mfcc)
            
            if feat is not None:
                X.append(feat)
                y.append(LABEL_MAP[parts[2]]) 
                total_sucessos += 1
        
        if len(X) > 0:
            np.save(os.path.join(target_folder, f"X_{name}.npy"), np.array(X))
            np.save(os.path.join(target_folder, f"y_{name}.npy"), np.array(y))
        else:
            print(f"ERRO CRÍTICO: Nenhum áudio extraído para o conjunto '{name}'.")
            return None
            
    print(f"Concluído! Total de áudios válidos convertidos: {total_sucessos}")
    return target_folder