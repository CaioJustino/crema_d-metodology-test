import os
import numpy as np
import librosa
import collections
import pandas as pd

SR = 16000
DURATION = 3.0
LABELS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

def get_splits_from_csv(csv_path):
    bad_files = {'FileName', '1040_ITH_SAD_XX', '1006_TIE_NEU_XX', '1013_WSI_DIS_XX', '1017_IWW_FEA_XX'}
    df = pd.read_csv(csv_path)
    valid_items = []
    for _, row in df.iterrows():
        wav_name = str(row.iloc[1]).replace('"', '').strip()
        if (not wav_name) or (wav_name in bad_files) or (wav_name == 'nan'): continue
        speaker_id = wav_name.split('_')[0]
        valid_items.append((wav_name, speaker_id))

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
    except: return None

def process_and_save(raw_path, base_processed_path, csv_path, n_mfcc):
    target_folder = f"{base_processed_path}_{n_mfcc}"
    if os.path.exists(target_folder):
        return target_folder

    print(f"Extraindo {n_mfcc} MFCCs...")
    splits = get_splits_from_csv(csv_path)
    os.makedirs(target_folder, exist_ok=True)

    for name, files in splits.items():
        X, y = [], []
        for wav in files:
            path = os.path.join(raw_path, f"{wav}.wav")
            if os.path.exists(path):
                parts = wav.split('_')
                feat = extract_mfcc(path, n_mfcc)
                if feat is not None:
                    X.append(feat)
                    y.append(LABEL_MAP[parts[2]])
        
        np.save(os.path.join(target_folder, f"X_{name}.npy"), np.array(X))
        np.save(os.path.join(target_folder, f"y_{name}.npy"), np.array(y))
    return target_folder