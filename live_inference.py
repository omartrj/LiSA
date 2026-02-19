import torch
import torchaudio
import numpy as np
import os
import argparse
import time
import pandas as pd
import csv
import matplotlib.pyplot as plt
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

class AudioStreamSimulator:
    def __init__(self, seq_dir):
        self.audio_data = []
        
        for i in range(1, NUM_MICS + 1):
            path = os.path.join(seq_dir, 'sound', f'microphone_{i}.wav')
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
    # gt.csv has 'time_s' column
    closest_idx = (gt_df['time_s'] - current_time).abs().idxmin()
    row = gt_df.iloc[closest_idx]
    return row['dist'], row['angle']

def generate_plots_and_statistics(csv_path, output_dir, postprocess_info=None):
    """
    Genera grafici e statistiche dal CSV di live inference.
    Identico a check_inference.py
    """
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    
    # Convert Polar (Dist, Angle) to Cartesian (X, Y)
    # Ground Truth
    gt_rad = np.deg2rad(df['gt_angle'])
    df['gt_x'] = df['gt_dist'] * np.cos(gt_rad)
    df['gt_y'] = df['gt_dist'] * np.sin(gt_rad)
    
    # Prediction
    pred_rad = np.deg2rad(df['pred_angle'])
    df['pred_x'] = df['pred_dist'] * np.cos(pred_rad)
    df['pred_y'] = df['pred_dist'] * np.sin(pred_rad)
    
    # Calcola errori
    df['error_dist'] = np.abs(df['gt_dist'] - df['pred_dist'])
    df['error_angle'] = np.abs((df['gt_angle'] - df['pred_angle'] + 180) % 360 - 180)
    
    # Metrics
    mae_dist = np.mean(df['error_dist'])
    
    # Angular error handling circularity
    diff_rad = np.arctan2(np.sin(pred_rad - gt_rad), np.cos(pred_rad - gt_rad))
    mae_angle = np.mean(np.abs(np.degrees(diff_rad)))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Plotting ---
    plt.figure(figsize=(15, 6))
    
    # 1. Spatial Trajectory (Top-Down View)
    plt.subplot(1, 2, 1)
    plt.plot(df['gt_x'], df['gt_y'], 'k--', label='Ground Truth', linewidth=2, alpha=0.6)
    plt.plot(df['pred_x'], df['pred_y'], 'r-', label='Prediction', linewidth=1.5, alpha=0.8)
    
    # Mark the Listener (0,0)
    plt.scatter(0, 0, c='green', marker='^', s=150, label='Listener', zorder=5)
    
    # Mark Start and End
    plt.scatter(df['gt_x'].iloc[0], df['gt_y'].iloc[0], c='blue', marker='o', label='Start')
    plt.scatter(df['gt_x'].iloc[-1], df['gt_y'].iloc[-1], c='black', marker='x', label='End')
    
    plt.title(f'Spatial Trajectory Reconstruction\n(MAE: {mae_dist:.1f}m, {mae_angle:.1f}°)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # 2. Time Series Analysis
    plt.subplot(2, 2, 2)
    plt.plot(df['time_s'], df['gt_dist'], 'k--', label='GT Dist')
    plt.plot(df['time_s'], df['pred_dist'], 'r-', label='Pred Dist')
    plt.ylabel('Distance [m]')
    plt.title('Distance over Time')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(df['time_s'], df['gt_angle'], 'k--', label='GT Angle')
    plt.plot(df['time_s'], df['pred_angle'], 'r-', label='Pred Angle')
    plt.ylabel('Angle [deg]')
    plt.xlabel('Time [s]')
    plt.title('Angle over Time')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-180, 180)
    plt.legend()
    
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, 'trajectory.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    
    # --- Statistics ---
    stats_lines = []
    stats_lines.append("="*60)
    stats_lines.append("LIVE INFERENCE STATISTICS")
    stats_lines.append("="*60)
    
    # Se postprocessing attivo, aggiungi info
    if postprocess_info and postprocess_info['enabled']:
        stats_lines.append(f"Post-processing:     ENABLED")
        stats_lines.append(f"  Method:            {postprocess_info['method']}")
        stats_lines.append(f"  History:           {postprocess_info['history']}")
        stats_lines.append("-"*60)
    
    stats_lines.append(f"Total Frames:        {len(df)}")
    stats_lines.append(f"Duration:            {df['time_s'].iloc[-1]:.2f} seconds")
    stats_lines.append(f"Avg Latency:         {df['latency_ms'].mean():.1f} ms")
    stats_lines.append(f"Max Latency:         {df['latency_ms'].max():.1f} ms")
    stats_lines.append("-"*60)
    stats_lines.append(f"Distance MAE:        {mae_dist:.2f} m")
    stats_lines.append(f"Distance Median:     {df['error_dist'].median():.2f} m")
    stats_lines.append(f"Distance RMSE:       {np.sqrt((df['error_dist']**2).mean()):.2f} m")
    
    # Errore percentuale sulla distanza (MAPE)
    mape_dist = (df['error_dist'] / df['gt_dist']).mean() * 100
    stats_lines.append(f"Distance MAPE:       {mape_dist:.1f}%")
    stats_lines.append("-"*60)
    stats_lines.append(f"Angle MAE:           {mae_angle:.2f}°")
    stats_lines.append(f"Angle Median:        {df['error_angle'].median():.2f}°")
    stats_lines.append(f"Angle RMSE:          {np.sqrt((df['error_angle']**2).mean()):.2f}°")
    stats_lines.append(f"Angle Acc <10°:      {(df['error_angle'] < 10).mean() * 100:.1f}%")
    stats_lines.append(f"Angle Acc <20°:      {(df['error_angle'] < 20).mean() * 100:.1f}%")
    stats_lines.append("="*60)
    
    stats_file = os.path.join(output_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write('\n'.join(stats_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=str, required=True)
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
    
    # Create output directory with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV Logger
    csv_path = os.path.join(output_dir, 'inference_log.csv')
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if has_gt:
        csv_writer.writerow(['time_s', 'pred_dist', 'gt_dist', 'pred_angle', 'gt_angle', 'latency_ms'])
    else:
        csv_writer.writerow(['time_s', 'pred_dist', 'pred_angle', 'latency_ms'])

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
                pred_dist, pred_angle, hidden_state = model(input_tensor, mic_coords, hidden_state)
            
            dist_m = pred_dist.item()
            sin_val = pred_angle[0, 0, 0].item()
            cos_val = pred_angle[0, 0, 1].item()
            angle_deg = np.degrees(np.arctan2(sin_val, cos_val))

            # Post-process if requested
            if args.postprocess:
                dist_m, angle_deg = tracker.update(dist_m, angle_deg)
            
            latency = (time.time() - start_proc) * 1000
            
            # Write to log
            if has_gt:
                # Get Ground Truth
                gt_d, gt_a = get_interpolated_gt(gt_df, timestamp)
                csv_writer.writerow([
                    f"{timestamp:.3f}", 
                    f"{dist_m:.3f}", 
                    f"{gt_d:.3f}", 
                    f"{angle_deg:.3f}", 
                    f"{gt_a:.3f}",
                    f"{latency:.1f}"
                ])
            else:
                csv_writer.writerow([
                    f"{timestamp:.3f}", 
                    f"{dist_m:.3f}", 
                    f"{angle_deg:.3f}",
                    f"{latency:.1f}"
                ])

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
        
        if has_gt:
            postprocess_info = {
                'enabled': args.postprocess,
                'method': args.smooth_method if args.postprocess else None,
                'history': args.history if args.postprocess else None
            }
            generate_plots_and_statistics(csv_path, output_dir, postprocess_info)
            
            print(f"Results saved to: {output_dir}/")
            print(f"  - inference_log.csv")
            print(f"  - trajectory.png")
            print(f"  - statistics.txt")
        else:
            print(f"Results saved to: {output_dir}/")
            print(f"  - inference_log.csv")

if __name__ == "__main__":
    main()