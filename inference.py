import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from model import LiSANet
from postprocessing import PostProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sequence(seq_name, processed_dir='data/processed'):
    """
    Carica una sequenza processata dal disco.
    """
    seq_path = os.path.join(processed_dir, f"{seq_name}.pt")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")
    
    data = torch.load(seq_path, mmap=False, weights_only=False)
    return data

def inference_on_sequence(model, data, device):
    """
    Esegue l'inferenza su tutti i frame di una sequenza.
    """
    # Estrai dati
    spectrograms = data['spectrograms'].to(device)  # (N, 8, F, T)
    gt_data = data['gt']  # (N, 3) -> [dist, sin(angle), cos(angle)]
    mic_coords = data['microphones'].unsqueeze(0).to(device)  # (1, 4, 3)
    
    num_frames = spectrograms.shape[0]
    
    # Storage per risultati
    pred_dists = []
    pred_angles = []
    
    # Hidden state della GRU (stateful)
    hidden_state = None
    
    model.eval()
    print(f"Running inference on {num_frames} frames...")
    
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            # Prepara input: (1, 1, 8, F, T)
            spec_frame = spectrograms[i].unsqueeze(0).unsqueeze(0)  # Batch=1, Seq=1
            
            # Forward pass: pred_angle è [1, 1, 2] con [:,:,0]=sin, [:,:,1]=cos
            pred_dist, pred_angle, hidden_state = model(spec_frame, mic_coords, hidden_state)
            
            # Estrai sin e cos per conversione in gradi
            pred_sin = pred_angle[0, 0, 0]
            pred_cos = pred_angle[0, 0, 1]
            
            # Converti sin/cos in angolo (gradi)
            pred_angle_rad = torch.atan2(pred_sin, pred_cos)
            pred_angle_deg = torch.rad2deg(pred_angle_rad)
            
            # Estrai valori
            pred_dists.append(pred_dist.item())
            pred_angles.append(pred_angle_deg.item())
    
    return np.array(pred_dists), np.array(pred_angles), gt_data.numpy()

def apply_postprocessing(pred_dists, pred_angles, method='median', history=5):
    """
    Applica post-processing (smoothing) alle predizioni.
    """
    print(f"Applying post-processing ({method}, history={history})...")
    processor = PostProcessor(history_length=history, method=method)
    
    smooth_dists = []
    smooth_angles = []
    
    for d, a in zip(pred_dists, pred_angles):
        sd, sa = processor.update(d, a)
        smooth_dists.append(sd)
        smooth_angles.append(sa)
    
    return np.array(smooth_dists), np.array(smooth_angles)

def save_predictions_csv(gt_data, pred_dists, pred_angles, output_path, sample_rate=20):
    """
    Salva le predizioni e il ground truth in un CSV.
    gt_data: (N, 3) con [dist, sin(angle), cos(angle)]
    """
    num_samples = len(pred_dists)
    timestamps = np.arange(num_samples) / sample_rate  # Tempo in secondi (20Hz)
    
    # Converti gt sin/cos in angoli (gradi)
    gt_angle_rad = np.arctan2(gt_data[:, 1], gt_data[:, 2])
    gt_angle_deg = np.rad2deg(gt_angle_rad)
    
    df = pd.DataFrame({
        'time_s': timestamps,
        'gt_dist': gt_data[:, 0],
        'gt_angle': gt_angle_deg,
        'pred_dist': pred_dists,
        'pred_angle': pred_angles,
        'error_dist': np.abs(gt_data[:, 0] - pred_dists),
        'error_angle': np.abs((gt_angle_deg - pred_angles + 180) % 360 - 180)
    })
    
    # Arrotonda a 2 decimali
    df = df.round(2)
    
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Predictions saved to: {output_path}")
    return df

def plot_trajectory(df, gt_data, pred_dists, pred_angles, output_path):
    """
    Crea un plot in stile check_inference.py con:
    - Traiettoria spaziale (top-down view)
    - Serie temporali di distanza e angolo
    """
    # Converti gt sin/cos in angoli (gradi)
    gt_angle_rad = np.arctan2(gt_data[:, 1], gt_data[:, 2])
    gt_angle_deg = np.rad2deg(gt_angle_rad)
    
    # Converti coordinate polari in cartesiane
    pred_rad = np.deg2rad(pred_angles)
    pred_x = pred_dists * np.cos(pred_rad)
    pred_y = pred_dists * np.sin(pred_rad)
    
    gt_rad = np.deg2rad(gt_angle_deg)
    gt_x = gt_data[:, 0] * np.cos(gt_rad)
    gt_y = gt_data[:, 0] * np.sin(gt_rad)
    
    # Calcola metriche
    mae_dist = np.mean(np.abs(gt_data[:, 0] - pred_dists))
    
    # Angular error handling circularity
    diff_rad = np.arctan2(np.sin(pred_rad - gt_rad), np.cos(pred_rad - gt_rad))
    mae_angle = np.mean(np.abs(np.degrees(diff_rad)))
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # 1. Spatial Trajectory (Top-Down View)
    plt.subplot(1, 2, 1)
    plt.plot(gt_x, gt_y, 'k--', label='Ground Truth', linewidth=2, alpha=0.6)
    plt.plot(pred_x, pred_y, 'r-', label='Prediction', linewidth=1.5, alpha=0.8)
    
    # Mark the listener (0,0)
    plt.scatter(0, 0, c='green', marker='^', s=150, label='Listener', zorder=5)
    
    # Mark Start and End
    plt.scatter(gt_x[0], gt_y[0], c='blue', marker='o', label='Start')
    plt.scatter(gt_x[-1], gt_y[-1], c='black', marker='x', label='End')
    
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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Trajectory check-style plot saved to: {output_path}")

def save_statistics(df, output_path, postprocess_info=None):
    """
    Salva statistiche di errore su file .txt e stampa a video.
    postprocess_info: dict con 'enabled', 'method', 'history' (opzionale)
    """
    # Calcola statistiche
    stats_lines = []
    stats_lines.append("="*60)
    stats_lines.append("INFERENCE STATISTICS")
    stats_lines.append("="*60)
    
    # Se postprocessing attivo, aggiungi info
    if postprocess_info and postprocess_info['enabled']:
        stats_lines.append(f"Post-processing:     ENABLED")
        stats_lines.append(f"  Method:            {postprocess_info['method']}")
        stats_lines.append(f"  History:           {postprocess_info['history']}")
        stats_lines.append("-"*60)
    
    stats_lines.append(f"Total Frames:        {len(df)}")
    stats_lines.append(f"Duration:            {df['time_s'].iloc[-1]:.2f} seconds")
    stats_lines.append("-"*60)
    stats_lines.append(f"Distance MAE:        {df['error_dist'].mean():.2f} m")
    stats_lines.append(f"Distance Median:     {df['error_dist'].median():.2f} m")
    stats_lines.append(f"Distance RMSE:       {np.sqrt((df['error_dist']**2).mean()):.2f} m")
    
    # Errore percentuale sulla distanza (MAPE)
    mape_dist = (df['error_dist'] / df['gt_dist']).mean() * 100
    stats_lines.append(f"Distance MAPE:       {mape_dist:.1f}%")
    stats_lines.append("-"*60)
    stats_lines.append(f"Angle MAE:           {df['error_angle'].mean():.2f}°")
    stats_lines.append(f"Angle Median:        {df['error_angle'].median():.2f}°")
    stats_lines.append(f"Angle RMSE:          {np.sqrt((df['error_angle']**2).mean()):.2f}°")
    stats_lines.append(f"Angle Acc <10°:      {(df['error_angle'] < 10).mean() * 100:.1f}%")
    stats_lines.append(f"Angle Acc <20°:      {(df['error_angle'] < 20).mean() * 100:.1f}%")
    stats_lines.append("="*60)
    
    # Salva su file
    with open(output_path, 'w') as f:
        f.write('\n'.join(stats_lines))
    
    print(f"Statistics saved to: {output_path}")

def main(args):
    # 1. Load Sequence
    print(f"Loading sequence: {args.seq}")
    data = load_sequence(args.seq, args.processed_dir)
    
    # 2. Load Model
    print(f"Loading model from: {args.model_path}")
    model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)
    
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 3. Run Inference
    pred_dists, pred_angles, gt_data = inference_on_sequence(model, data, DEVICE)
    
    # 4. Apply Post-Processing (if enabled)
    postprocess_info = {
        'enabled': args.postprocess,
        'method': args.smooth_method if args.postprocess else None,
        'history': args.history if args.postprocess else None
    }
    
    if args.postprocess:
        pred_dists, pred_angles = apply_postprocessing(
            pred_dists, pred_angles, 
            method=args.smooth_method, 
            history=args.history
        )
    
    # 5. Create Output Directory
    output_dir = os.path.join(args.output_dir, args.seq)
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Save Predictions CSV
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df = save_predictions_csv(gt_data, pred_dists, pred_angles, csv_path, sample_rate=20)
    
    # 7. Plot Trajectory
    plot_check_path = os.path.join(output_dir, 'trajectory.png')
    plot_trajectory(df, gt_data, pred_dists, pred_angles, plot_check_path)
    
    # 8. Save Statistics
    stats_path = os.path.join(output_dir, 'statistics.txt')
    save_statistics(df, stats_path, postprocess_info)
    
    print(f"\nAll outputs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a single sequence and visualize results."
    )
    parser.add_argument("--seq", type=str, required=True, help="Sequence name (e.g., seq000)")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="inference_results")

    # Post-processing options
    parser.add_argument("--postprocess", action='store_true')
    parser.add_argument("--smooth_method", type=str, default="median", choices=['mean', 'median'], help="Smoothing method (median or mean)")
    parser.add_argument("--history", type=int, default=11, help="Smoothing window size")
    
    args = parser.parse_args()
    main(args)
