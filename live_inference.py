import torch
import torchaudio
import numpy as np
import os
import argparse
import time
import pandas as pd
import csv
import matplotlib.pyplot as plt
import signal
from datetime import datetime
from model import LiSANet
from postprocessing import PostProcessor

# Stesse costanti di preprocessing per coerenza con training/inference
AUDIO_SAMPLE_RATE = 48000
CONTEXT_WINDOW_MS = 100
UPDATE_INTERVAL_MS = 50
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 1024
MAX_FREQ_BINS = 128
NUM_MICS = 4

# DATA GENERATION FILE
GLOBAL_MAX_PEAK = 25000.0 # Osservato empiricamente
WARMUP_FRAMES = 10        # Frame iniziali usati solo per scaldare la GRU, output scartato

class AudioStreamSimulator:

    #WAV_FILE_PREFIX = 'microphone_'
    WAV_FILE_PREFIX = 'mic'

    def __init__(self, seq_dir):
        self.audio_data = []
        
        for i in range(1, NUM_MICS + 1):
            path = os.path.join(seq_dir, 'sound', f'{self.WAV_FILE_PREFIX}{i}.wav')
            waveform, sr = torchaudio.load(path)
            if sr != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)
            self.audio_data.append(waveform[0])
            
        self.audio_data = torch.stack(self.audio_data)
        self.current_idx = 0
        self.total_samples = self.audio_data.shape[1]
        self.chunk_size = int((UPDATE_INTERVAL_MS / 1000.0) * AUDIO_SAMPLE_RATE)
        
    def get_next_chunk(self):
        # Calculate current timestamp before updating index
        timestamp = self.current_idx / AUDIO_SAMPLE_RATE
        
        end_idx = self.current_idx + self.chunk_size
        
        if end_idx > self.total_samples:
            remaining = self.total_samples - self.current_idx
            part1 = self.audio_data[:, self.current_idx:]
            part2 = self.audio_data[:, :self.chunk_size - remaining]
            chunk = torch.cat([part1, part2], dim=1)
            self.current_idx = self.chunk_size - remaining
            # Reset timestamp if looped (optional, depends on needs)
        else:
            chunk = self.audio_data[:, self.current_idx:end_idx]
            self.current_idx = end_idx
            
        return chunk, timestamp

class OnlinePreprocessor:
    def __init__(self):
        self.buffer_size = int((CONTEXT_WINDOW_MS / 1000.0) * AUDIO_SAMPLE_RATE)
        self.buffer = torch.zeros(NUM_MICS, self.buffer_size)
        self.window_fn = torch.hann_window(WIN_LENGTH)
        
    def process(self, new_chunk):
        chunk_len = new_chunk.shape[1]
        self.buffer = torch.roll(self.buffer, -chunk_len, dims=1)
        self.buffer[:, -chunk_len:] = new_chunk
        
        stft = torch.stft(
            self.buffer,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window_fn,
            return_complex=True,
            center=False,
            normalized=False
        )
        
        stft = stft[:, :MAX_FREQ_BINS, :]
        spec = torch.cat([stft.real, stft.imag], dim=0)
        return spec.unsqueeze(0).unsqueeze(0)

def get_interpolated_gt(gt_df, current_time):
    # Find the row with the closest timestamp
    closest_idx = (gt_df['time_s'] - current_time).abs().idxmin()
    row = gt_df.iloc[closest_idx]
    is_active = int(row['is_active']) if 'is_active' in row else 0
    return row['dist'], row['angle'], is_active

def generate_plots_and_statistics(csv_path, output_dir, has_gt=True, postprocess_info=None):
    """
    Genera grafici e statistiche dal CSV di live inference.
    Se has_gt=False, genera solo plot della predizione senza comparazioni.
    """
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)

    # Maschere attività
    pred_active = (df['pred_active_prob'] >= 0.5).values if 'pred_active_prob' in df.columns else np.ones(len(df), dtype=bool)

    # Convert Polar (Dist, Angle) to Cartesian (X, Y)
    pred_rad = np.deg2rad(df['pred_angle'])
    df['pred_x'] = df['pred_dist'] * np.cos(pred_rad)
    df['pred_y'] = df['pred_dist'] * np.sin(pred_rad)
    
    if has_gt:
        gt_active = df['gt_active'].astype(bool).values if 'gt_active' in df.columns else np.ones(len(df), dtype=bool)
        gt_rad = np.deg2rad(df['gt_angle'])
        df['gt_x'] = df['gt_dist'] * np.cos(gt_rad)
        df['gt_y'] = df['gt_dist'] * np.sin(gt_rad)
        
        # Calcola errori solo su frame attivi
        active_mask = gt_active
        if active_mask.any():
            err_dist   = np.abs(df['gt_dist'].values[active_mask] - df['pred_dist'].values[active_mask])
            diff_rad_a = np.arctan2(np.sin(pred_rad.values[active_mask] - gt_rad.values[active_mask]),
                                    np.cos(pred_rad.values[active_mask] - gt_rad.values[active_mask]))
            mae_dist  = np.mean(err_dist)
            mae_angle = np.mean(np.abs(np.degrees(diff_rad_a)))
        else:
            mae_dist = mae_angle = float('nan')
        
        # Detection accuracy
        pred_active_bin = (df['pred_active_prob'].values >= 0.5).astype(int)
        det_acc = np.mean(pred_active_bin == gt_active.astype(int)) * 100
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Spatial Trajectory
    ax = axes[0, 0]
    if has_gt:
        gt_x_arr = df['gt_x'].values
        gt_y_arr = df['gt_y'].values
        gt_x_inact = np.where(~gt_active, gt_x_arr, np.nan)
        gt_y_inact = np.where(~gt_active, gt_y_arr, np.nan)
        gt_x_act   = np.where(gt_active,  gt_x_arr, np.nan)
        gt_y_act   = np.where(gt_active,  gt_y_arr, np.nan)
        ax.plot(gt_x_inact, gt_y_inact, color='lightgrey', linewidth=2, label='GT (inactive)')
        ax.plot(gt_x_act,   gt_y_act,   'k--', linewidth=2, alpha=0.8, label='GT (active)')
        ax.set_title(f'Spatial Trajectory\n(MAE: {mae_dist:.1f}m, {mae_angle:.1f}° | Det Acc: {det_acc:.1f}%)')
    else:
        ax.set_title('Predicted Source Trajectory')

    pred_x_act = np.where(pred_active, df['pred_x'].values, np.nan)
    pred_y_act = np.where(pred_active, df['pred_y'].values, np.nan)
    ax.plot(pred_x_act, pred_y_act, 'r-', linewidth=1.5, alpha=0.8, label='Prediction (active)')
    ax.scatter(0, 0, c='green', marker='^', s=150, label='Listener', zorder=5)
    ax.scatter(df['pred_x'].iloc[0], df['pred_y'].iloc[0], c='blue', marker='o', label='Start')
    ax.scatter(df['pred_x'].iloc[-1], df['pred_y'].iloc[-1], c='black', marker='x', label='End')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 2. Distance over Time
    ax = axes[0, 1]
    if has_gt:
        gt_dist_inact = np.where(~gt_active, df['gt_dist'], np.nan)
        gt_dist_act   = np.where(gt_active,  df['gt_dist'], np.nan)
        ax.plot(df['time_s'], gt_dist_inact, color='lightgrey', linewidth=2, label='GT Dist (inactive)')
        ax.plot(df['time_s'], gt_dist_act,   'k--', linewidth=2, alpha=0.8, label='GT Dist (active)')
    pred_dist_act = np.where(pred_active, df['pred_dist'], np.nan)
    ax.plot(df['time_s'], pred_dist_act, 'r-', label='Pred Dist (active)')
    ax.set_ylabel('Distance [m]'); ax.set_title('Distance over Time')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend()
    
    # 3. Angle over Time
    ax = axes[1, 1]
    if has_gt:
        gt_angle_inact = np.where(~gt_active, df['gt_angle'], np.nan)
        gt_angle_act   = np.where(gt_active,  df['gt_angle'], np.nan)
        ax.plot(df['time_s'], gt_angle_inact, color='lightgrey', linewidth=2, label='GT Angle (inactive)')
        ax.plot(df['time_s'], gt_angle_act,   'k--', linewidth=2, alpha=0.8, label='GT Angle (active)')
    pred_angle_act = np.where(pred_active, df['pred_angle'], np.nan)
    ax.plot(df['time_s'], pred_angle_act, 'r-', label='Pred Angle (active)')
    ax.set_ylabel('Angle [deg]'); ax.set_xlabel('Time [s]')
    ax.set_title('Angle over Time')
    ax.set_ylim(-180, 180); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 4. Siren Detection
    ax = axes[1, 0]
    if has_gt:
        ax.step(df['time_s'], gt_active.astype(int), 'k--', where='post', label='GT Active', linewidth=2)
    if 'pred_active_prob' in df.columns:
        ax.plot(df['time_s'], df['pred_active_prob'], 'b-', label='Pred Prob', alpha=0.7)
        ax.axhline(0.5, color='orange', linestyle=':', label='Threshold 0.5')
    ax.set_ylabel('Active'); ax.set_xlabel('Time [s]')
    ax.set_title('Siren Detection')
    ax.set_ylim(-0.05, 1.05); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    plt.tight_layout()
    out_file = os.path.join(output_dir, 'trajectory.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    
    # --- Statistics ---
    if has_gt:
        stats_lines = []
        stats_lines.append("="*60)
        stats_lines.append("LIVE INFERENCE STATISTICS")
        stats_lines.append("="*60)
        
        if postprocess_info and postprocess_info['enabled']:
            stats_lines.append(f"Post-processing:     ENABLED")
            stats_lines.append(f"  Method:            {postprocess_info['method']}")
            stats_lines.append(f"  History:           {postprocess_info['history']}")
            stats_lines.append("-"*60)
        
        stats_lines.append(f"Total Frames:        {len(df)}")
        stats_lines.append(f"Duration:            {df['time_s'].iloc[-1]:.2f} seconds")
        stats_lines.append(f"Active Frames (GT):  {int(gt_active.sum())} ({gt_active.mean()*100:.1f}%)")
        stats_lines.append(f"Avg Latency:         {df['latency_ms'].mean():.1f} ms")
        stats_lines.append(f"Max Latency:         {df['latency_ms'].max():.1f} ms")
        stats_lines.append("-"*60)
        stats_lines.append(f"Detection Accuracy:  {det_acc:.1f}%")
        stats_lines.append("-"*60)
        if active_mask.any():
            err_dist_col = np.abs(df['gt_dist'].values[active_mask] - df['pred_dist'].values[active_mask])
            stats_lines.append(f"Distance MAE:        {mae_dist:.2f} m  (active frames only)")
            stats_lines.append(f"Distance Median:     {np.median(err_dist_col):.2f} m")
            stats_lines.append(f"Distance RMSE:       {np.sqrt(np.mean(err_dist_col**2)):.2f} m")
            mape_dist = (err_dist_col / (df['gt_dist'].values[active_mask] + 1e-6)).mean() * 100
            stats_lines.append(f"Distance MAPE:       {mape_dist:.1f}%")
            stats_lines.append("-"*60)
            err_angle_col = np.abs(np.degrees(diff_rad_a))
            stats_lines.append(f"Angle MAE:           {mae_angle:.2f}°  (active frames only)")
            stats_lines.append(f"Angle Median:        {np.median(err_angle_col):.2f}°")
            stats_lines.append(f"Angle RMSE:          {np.sqrt(np.mean(err_angle_col**2)):.2f}°")
            stats_lines.append(f"Angle Acc <10°:      {(err_angle_col < 10).mean() * 100:.1f}%")
            stats_lines.append(f"Angle Acc <20°:      {(err_angle_col < 20).mean() * 100:.1f}%")
        else:
            stats_lines.append("No active frames in this sequence.")
        stats_lines.append("="*60)
        
        stats_file = os.path.join(output_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            f.write('\n'.join(stats_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=str, required=True)
    # La `seq_dir` deve contenere:
    # - una cartella `sound/` con i file `microphone_1.wav`, ..., `microphone_4.wav`
    # - un file `microphones.csv` con le coordinate dei microfoni
    # - un file `gt.csv` con la ground truth (opzionale, per valutazione)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="live_inference_results")
    
    # Post-processing options
    parser.add_argument("--postprocess", action='store_true')
    parser.add_argument("--smooth_method", type=str, default="median", choices=['mean', 'median'], help="Smoothing method (median or mean)")
    parser.add_argument("--history", type=int, default=11, help="Smoothing window size")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Microphones
    mic_path = os.path.join(args.seq_dir, 'microphones.csv')
    mics_df = pd.read_csv(mic_path)
    mic_coords = torch.tensor(mics_df[['mx', 'my', 'mz']].values, dtype=torch.float32).to(device)
    mic_coords = mic_coords.unsqueeze(0)

    # Load Ground Truth (optional)
    gt_path = os.path.join(args.seq_dir, 'gt.csv')
    has_gt = os.path.exists(gt_path)
    
    if has_gt:
        gt_df = pd.read_csv(gt_path)
        print("Ground truth found - evaluation mode enabled")
    else:
        gt_df = None
        print("No ground truth found - prediction-only mode")

    # Load Model
    model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    streamer = AudioStreamSimulator(args.seq_dir)
    preprocessor = OnlinePreprocessor()
    hidden_state = None
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create output directory with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV Logger
    csv_path = os.path.join(output_dir, 'inference_log.csv')
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if has_gt:
        csv_writer.writerow(['time_s', 'pred_dist', 'gt_dist', 'pred_angle', 'gt_angle',
                             'pred_active_prob', 'gt_active', 'latency_ms'])
    else:
        csv_writer.writerow(['time_s', 'pred_dist', 'pred_angle', 'pred_active_prob', 'latency_ms'])

    print(f"Logging to {csv_path}...")
    print(f"Output directory: {output_dir}")
    
    print("Press Ctrl+C to stop...")

    # Durata target di ogni ciclo in secondi (es. 0.05s)
    target_loop_time = UPDATE_INTERVAL_MS / 1000.0
    
    # Imposta il "clock" iniziale
    next_deadline = time.time() + target_loop_time

    # Iniziallizza il post-processor se richiesto
    if args.postprocess:
        tracker = PostProcessor(history_length=args.history, method=args.smooth_method)
        print(f"Post-processing enabled: {args.smooth_method}, history={args.history}")

    frame_idx = 0  # contatore frame per warm-up

    try:
        # Run for the length of the file (stop at loop)
        while True:
            start_proc = time.time()
            
            chunk, timestamp = streamer.get_next_chunk()
            
            # Stop if we looped (timestamp reset) or exceeded max time
            if timestamp > streamer.total_samples / AUDIO_SAMPLE_RATE:
                break
            
            input_tensor = preprocessor.process(chunk).to(device)
            
            with torch.no_grad():
                pred_dist, pred_angle, pred_active_logit, hidden_state = model(input_tensor, mic_coords, hidden_state)

            # Warm-up: aggiorna la hidden state ma scarta l'output
            frame_idx += 1
            if frame_idx <= WARMUP_FRAMES:
                next_deadline += target_loop_time
                continue
            
            dist_m = pred_dist.item()
            sin_val = pred_angle[0, 0, 0].item()
            cos_val = pred_angle[0, 0, 1].item()
            angle_deg = np.degrees(np.arctan2(sin_val, cos_val))
            pred_active_prob = torch.sigmoid(pred_active_logit[0, 0]).item()

            # Post-process if requested
            if args.postprocess:
                dist_m, angle_deg = tracker.update(dist_m, angle_deg)
            
            latency = (time.time() - start_proc) * 1000
            
            # Write to log
            if has_gt:
                # Get Ground Truth
                gt_d, gt_a, gt_active = get_interpolated_gt(gt_df, timestamp)
                csv_writer.writerow([
                    f"{timestamp:.3f}",
                    f"{dist_m:.3f}",
                    f"{gt_d:.3f}",
                    f"{angle_deg:.3f}",
                    f"{gt_a:.3f}",
                    f"{pred_active_prob:.4f}",
                    f"{gt_active}",
                    f"{latency:.1f}"
                ])
            else:
                csv_writer.writerow([
                    f"{timestamp:.3f}",
                    f"{dist_m:.3f}",
                    f"{angle_deg:.3f}",
                    f"{pred_active_prob:.4f}",
                    f"{latency:.1f}"
                ])
            
            # Flush CSV periodically to ensure data is written
            csv_file.flush()

            # 2. GESTIONE TEMPO REALE
            now = time.time()
            time_to_wait = next_deadline - now
            
            if time_to_wait > 0:
                # Se siamo in anticipo, dormiamo il giusto per sincronizzarci
                time.sleep(time_to_wait)
            else:
                # Se time_to_wait è negativo, significa che il modello è LENTO!
                # Stiamo violando il vincolo real-time.
                lag = abs(time_to_wait) * 1000
                print(f"WARNING: System lag! Processing took too long. Behind by {lag:.1f}ms")
                
                # Opzionale: Resettiamo la deadline per non accumulare ritardo sul prossimo frame
                # (Skip frame logic)
                next_deadline = time.time()

            # Imposta la sveglia per il prossimo frame
            next_deadline += target_loop_time
            
            # Optional: Print every 10 steps to avoid clutter
            if int(timestamp * 1000) % 500 == 0:  # Every 0.5 seconds
                print(f"T: {timestamp:.2f}s | Dist: {dist_m:.1f}m | Processing Load: {(1 - time_to_wait/target_loop_time)*100:.0f}%")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        csv_file.close()
        
        # Generate plots (always) and statistics (only if has_gt)
        postprocess_info = {
            'enabled': args.postprocess,
            'method': args.smooth_method if args.postprocess else None,
            'history': args.history if args.postprocess else None
        }
        generate_plots_and_statistics(csv_path, output_dir, has_gt, postprocess_info)
        
        print(f"Results saved to: {output_dir}/")
        print(f"  - inference_log.csv")
        print(f"  - trajectory.png")
        if has_gt:
            print(f"  - statistics.txt")

if __name__ == "__main__":
    main()