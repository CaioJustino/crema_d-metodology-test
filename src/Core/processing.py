import os
import numpy as np
import librosa
import collections
import pandas as pd

# Configurações de áudio
SR = 16000
DURATION = 3.0
N_MFCC = 40
LABELS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

def get_splits_from_csv(csv_path):
    """Lógica fiel ao crema_d.py usando o summaryTable.csv"""
    bad_files = {
        'FileName',
        '1040_ITH_SAD_XX',
        '1006_TIE_NEU_XX',
        '1013_WSI_DIS_XX',
        '1017_IWW_FEA_XX',
    }

    # Lemos o CSV sem assumir nomes de colunas específicos
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")
        return {}

    valid_items = []
    for _, row in df.iterrows():
        try:
            # .iloc[1] pega a segunda coluna (onde geralmente está o nome do arquivo)
            wav_name = str(row.iloc[1]).replace('"', '').strip()
            
            if (not wav_name) or (wav_name in bad_files) or (wav_name == 'nan'):
                continue
                
            speaker_id = wav_name.split('_')[0]
            valid_items.append((wav_name, speaker_id))
        except Exception:
            continue

    # Lógica de Split por Locutor
    groups = sorted(set(group_id for _, group_id in valid_items))
    rng = np.random.RandomState(0) 
    rng.shuffle(groups)

    split_probs = [('train', 0.7), ('validation', 0.1), ('test', 0.2)]
    n_items = len(groups)
    
    split_boundaries = []
    sum_p = 0.0
    for name, p in split_probs:
        prev = sum_p
        sum_p += p
        split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))
    
    split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1], n_items)

    group_id_to_split = {}
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            group_id_to_split[groups[i]] = split_name

    splits = collections.defaultdict(list)
    for wav_name, group_id in valid_items:
        if group_id in group_id_to_split:
            split = group_id_to_split[group_id]
            splits[split].append(wav_name)
    
    return splits

def extract_mfcc(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        if len(audio) < SR * DURATION:
            audio = np.pad(audio, (0, int(SR * DURATION - len(audio))))
        elif len(audio) > SR * DURATION:
            audio = audio[:int(SR * DURATION)]
            
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        return mfcc.T 
    except Exception:
        return None

def process_and_save(raw_path, processed_path, csv_path):
    splits = get_splits_from_csv(csv_path)
    
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    for split_name, file_list in splits.items():
        print(f"Processando split: {split_name} ({len(file_list)} arquivos)...")
        X, y = [], []
        
        for wav_name in file_list:
            file_path = os.path.join(raw_path, f"{wav_name}.wav")
            
            if os.path.exists(file_path):
                # Extrai a label do nome do arquivo (ex: 1001_DFA_ANG_XX -> ANG)
                parts = wav_name.split('_')
                if len(parts) >= 3:
                    label_str = parts[2]
                    if label_str in LABEL_MAP:
                        mfcc = extract_mfcc(file_path)
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(LABEL_MAP[label_str])
        
        if X:
            np.save(os.path.join(processed_path, f"X_{split_name}.npy"), np.array(X))
            np.save(os.path.join(processed_path, f"y_{split_name}.npy"), np.array(y))
        else:
            print(f"Aviso: Nenhum dado processado para o split {split_name}")