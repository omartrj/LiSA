"""
Questo file è pensato per essere eseguito una sola volta, prima di far partire il training.
Il suo scopo è quello di processare le sequenze grezze che si trovano in `data/raw` e di salvarle in `data/processed` in un formato più comodo per il training.
Ogni sequenza *raw* è composta da:
- 4 file audio (dentro la cartella `sound`), uno per ogni microfono, in formato .wav
- `gt.csv`, con header `time_s,sx,sy,dist,angle`, che contiene le coordinate (sx, sy), la distanza e l'angolo della sorgente sonora rispetto all'origine del sistema di riferimento, campionati ogni 0.05 secondi (20Hz)
- `microphones.csv`, con header `mic_id,mx,my,mz`, che contiene le coordinate (mx, my, mz) di ogni microfono, in metri, rispetto all'origine del sistema di riferimento.
In particolare, il processamento consiste in:
- Caricare i 4 file audio
- Caricare i file gt.csv e microphones.csv
- Per ogni campione di gt.csv:
    * Estrarre gli **ultimi** CONTEXT_WINDOW_MS millisecondi di audio da ogni microfono. Se non sono disponibili (inizio della registrazione), scarto il campione
    * Calcolare lo spettrogramma complesso di ogni finestra audio -> ottengo 4 spettrogrammi complessi
    * Dato theta (l'angolo in gradi in gt.csv), calcolare sin(theta) e cos(theta) e salvarli come target invece di theta, per evitare problemi di discontinuità circolare
- Una volta processati i campioni di una sequenza, costruire un dizionario con le seguenti chiavi:
    * `spectrograms`: Tensor di forma (N, 8, F, T), dove N è il numero di campioni in gt.csv, 8 è il numero di canali complessi (4 microfoni x 2 canali: reale e immaginario), F è il numero di bande di frequenza dello spettrogramma, T è il numero di frame temporali dello spettrogramma
    * `gt`: Tensor di forma (N, 4), con colonne (distance, sin(angle), cos(angle), is_active)
    * `microphones`: Tensor di forma (NUM_MICROPHONES, 3), contenente le coordinate (x, y, z) di ogni microfono
- Salvare il dizionario in un file `.pt`.
"""
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import json
from tqdm import tqdm

# COSTANTI
AUDIO_SAMPLE_RATE = 48000   # Target Sample Rate
GT_SAMPLE_RATE = 20         # Frequenza aggiornamento ground truth (50ms)
CONTEXT_WINDOW_MS = 100     # Finestra audio passata al modello (100ms)
NUM_MICROPHONES = 4

# Parametri STFT (Devono coincidere con quelli usati in inference/training)
N_FFT = 1024                # Numero di campioni per finestra (21.3ms a 48kHz) 
WIN_LENGTH = 1024           # Lunghezza della finestra (stesso di N_FFT per finestre non sovrapposte)
HOP_LENGTH = 160            # Passo di avanzamento (160 campioni = 3.3ms a 48kHz, per avere ~30 frame per finestra di 100ms)

# Ottimizzazione
MAX_FREQ_BINS = 128 # 128 bin * (48000Hz / 1024) = ~6000Hz. Copre fondamentali e armoniche della sirena (500Hz-4kHz), scartando frequenze inutili

# Percorsi
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
AUDIO_SUBDIR = 'sound'
GT_FILE = 'gt.csv'
MIC_FILE = 'microphones.csv'


def process_sequence(seq_name):
    seq_path = os.path.join(RAW_DATA_DIR, seq_name)
    audio_path = os.path.join(seq_path, AUDIO_SUBDIR)
    
    # Carica ground truth e posizioni microfoni
    try:
        gt_df = pd.read_csv(os.path.join(seq_path, GT_FILE))
        mics_df = pd.read_csv(os.path.join(seq_path, MIC_FILE))
    except FileNotFoundError as e:
        print(f"Skipping {seq_name}: Metadata not found ({e})")
        return

    mic_coords = torch.tensor(mics_df[['mx', 'my', 'mz']].values, dtype=torch.float32)

    # Carica i 4 file audio dei microfoni
    waveforms = []
    for i in range(1, NUM_MICROPHONES + 1):
        af = os.path.join(audio_path, f'microphone_{i}.wav')
        if not os.path.exists(af):
            print(f"Skipping {seq_name}: {af} not found.")
            return
        waveform, sr = torchaudio.load(af)
        # Assicurati che sia float tra -1 e 1
        if waveform.abs().max() > 1.0:
            waveform = waveform.float() / 32768.0
        waveforms.append(waveform[0]) # Prendiamo solo il canale mono

    # Stack: (4, Total_Samples)
    full_audio = torch.stack(waveforms)
    
    # Finestra di Hann per la STFT
    window_fn = torch.hann_window(WIN_LENGTH)

    # Quanti campioni audio servono per 100ms?
    samples_per_window = int((CONTEXT_WINDOW_MS / 1000.0) * AUDIO_SAMPLE_RATE) # 4800 campioni per 100ms a 48kHz
    
    spectrogram_list = []
    gt_list = []

    # Itera i campioni di gt.csv e processa l'audio corrispondente
    for _, row in gt_df.iterrows():
        t_end_sec = row['time_s']
        
        # Indici nel file audio
        end_idx = int(t_end_sec * AUDIO_SAMPLE_RATE)
        start_idx = end_idx - samples_per_window

        # Se siamo all'inizio della registrazione e non abbiamo 100ms di storico, saltiamo
        if start_idx < 0:
            continue
            
        # Controllo fine file
        if end_idx > full_audio.shape[1]:
            continue
        
        chunk = full_audio[:, start_idx:end_idx]

        # Altro check sulla dimensione (per evitare errori STFT)
        if chunk.shape[1] != samples_per_window:
            continue

        # Calcola STFT
        # Input: (4, Samples) -> Output: (4, Freq, Time, Complex)
        stft = torch.stft(
            chunk,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window_fn,
            return_complex=True,
            center=False, 
            normalized=False
        )

        # Filtro passa basso:
        # Tagliamo le frequenze sopra i 6kHz (inutili per la sirena) riducendo l'input del 75% (513->128).
        # Questo velocizza drasticamente la CNN e rimuove rumore ad alta frequenza.
        stft = stft[:, :MAX_FREQ_BINS, :]

        # Prepara input per la rete: Concatena Reale e Immaginario
        # stft.real: (4, F, T)
        # stft.imag: (4, F, T)
        # Output Spec: (8, F, T)
        spec_tensor = torch.cat([stft.real, stft.imag], dim=0)
        
        # Estrae i target di distanza, angolo e attività
        dist = row['dist']
        angle_deg = row['angle']
        is_active = float(row['is_active'])

        # Calcolo sin(theta) e cos(theta)
        angle_rad = np.deg2rad(angle_deg)
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)
        
        # (N, 4) con colonne (distance, sin(angle), cos(angle), is_active)
        gt_tensor = torch.tensor([dist, sin_angle, cos_angle, is_active], dtype=torch.float32)

        spectrogram_list.append(spec_tensor)
        gt_list.append(gt_tensor)

    # Salva
    if len(spectrogram_list) > 0:
        final_specs = torch.stack(spectrogram_list) # Shape: (N, 8, F, T)
        final_gt = torch.stack(gt_list)             # Shape: (N, 4)
        
        processed_dict = {
            "spectrograms": final_specs,
            "gt": final_gt,
            "microphones": mic_coords,
        }

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DATA_DIR, f"{seq_name}.pt")
        torch.save(processed_dict, out_path)

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Directory {RAW_DATA_DIR} not found. Run simulation first.")

    sequences = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    sequences.sort()
    
    print(f"Found {len(sequences)} sequences to process.")
    print("Starting preprocessing...\n")
    
    for seq in tqdm(sequences, desc="Processing sequences", unit="seq"):
        process_sequence(seq)
    
    # Metadata utile per il DataLoader
    print("\nGenerating metadata cache...")
    metadata = {}
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.pt')]
    
    for pt_file in processed_files:
        pt_path = os.path.join(PROCESSED_DATA_DIR, pt_file)
        data = torch.load(pt_path, weights_only=True)
        metadata[pt_file] = data['gt'].shape[0]
    
    meta_path = os.path.join(PROCESSED_DATA_DIR, 'metadata_cache.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata cache saved: {len(metadata)} sequences -> {meta_path}")
