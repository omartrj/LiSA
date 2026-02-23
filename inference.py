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
WARMUP_FRAMES = 10  # Frame iniziali usati solo per scaldare la GRU, output scartato

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
    gt_data = data['gt']  # (N, 4) -> [dist, sin(angle), cos(angle), is_active]
    mic_coords = data['microphones'].unsqueeze(0).to(device)  # (1, 4, 3)
    
    num_frames = spectrograms.shape[0]
    
    # Storage per risultati
    pred_dists = []
    pred_angles = []
    pred_actives = []
    
    # Hidden state della GRU (stateful)
    hidden_state = None
    
    model.eval()
    print(f"Running inference on {num_frames} frames...")
    
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            # Prepara input: (1, 1, 8, F, T)
            spec_frame = spectrograms[i].unsqueeze(0).unsqueeze(0)  # Batch=1, Seq=1
            
            # Forward pass
            pred_dist, pred_angle, pred_active_logit, hidden_state = model(spec_frame, mic_coords, hidden_state)

            # Warm-up: aggiorna la hidden state ma scarta l'output
            if i < WARMUP_FRAMES:
                continue
            
            # Attività: sigmoid del logit -> probabilità [0, 1]
            pred_active_prob = torch.sigmoid(pred_active_logit[0, 0]).item()
            
            # Estrai sin e cos per conversione in gradi
            pred_sin = pred_angle[0, 0, 0]
            pred_cos = pred_angle[0, 0, 1]
            
            # Converti sin/cos in angolo (gradi)
            pred_angle_rad = torch.atan2(pred_sin, pred_cos)
            pred_angle_deg = torch.rad2deg(pred_angle_rad)
            
            # Estrai valori
            pred_dists.append(pred_dist.item())
            pred_angles.append(pred_angle_deg.item())
            pred_actives.append(pred_active_prob)
    
    return np.array(pred_dists), np.array(pred_angles), np.array(pred_actives), gt_data[WARMUP_FRAMES:].numpy()

def apply_postprocessing(pred_dists, pred_angles, pred_actives, method='median', history=5):
    """
    Applica post-processing (smoothing) alle predizioni di distanza e angolo.
    La probabilità di attività non viene smoothata (già stabile grazie alla GRU).
    """
    print(f"Applying post-processing ({method}, history={history})...")
    processor = PostProcessor(history_length=history, method=method)
    
    smooth_dists = []
    smooth_angles = []
    
    for d, a in zip(pred_dists, pred_angles):
        sd, sa = processor.update(d, a)
        smooth_dists.append(sd)
        smooth_angles.append(sa)
    
    return np.array(smooth_dists), np.array(smooth_angles), pred_actives

def save_predictions_csv(gt_data, pred_dists, pred_angles, pred_actives, output_path, sample_rate=20):
    """
    Salva le predizioni e il ground truth in un CSV.
    gt_data: (N, 4) con [dist, sin(angle), cos(angle), is_active]
    """
    num_samples = len(pred_dists)
    # I timestamp partono dopo il warm-up
    timestamps = (np.arange(num_samples) + WARMUP_FRAMES) / sample_rate
    
    # Converti gt sin/cos in angoli (gradi)
    gt_angle_rad = np.arctan2(gt_data[:, 1], gt_data[:, 2])
    gt_angle_deg = np.rad2deg(gt_angle_rad)
    
    gt_active = gt_data[:, 3].astype(int)
    pred_active_binary = (pred_actives >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'time_s': timestamps,
        'gt_dist': gt_data[:, 0],
        'gt_angle': gt_angle_deg,
        'gt_active': gt_active,
        'pred_dist': pred_dists,
        'pred_angle': pred_angles,
        'pred_active_prob': pred_actives,
        'pred_active': pred_active_binary,
        'error_dist': np.abs(gt_data[:, 0] - pred_dists),
        'error_angle': np.abs((gt_angle_deg - pred_angles + 180) % 360 - 180),
        'error_active': (gt_active != pred_active_binary).astype(int)
    })
    
    # Arrotonda a 2 decimali
    df = df.round(2)
    
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Predictions saved to: {output_path}")
    return df

def plot_trajectory(df, gt_data, pred_dists, pred_angles, output_path, mic_coords=None):
    """
    Crea un plot con:
    - Traiettoria spaziale (top-down view)
    - Serie temporali di distanza, angolo e rilevamento sirena
    """
    # Converti gt sin/cos in angoli (gradi)
    gt_angle_rad = np.arctan2(gt_data[:, 1], gt_data[:, 2])
    gt_angle_deg = np.rad2deg(gt_angle_rad)
    gt_active = gt_data[:, 3].astype(bool)
    pred_active = df['pred_active'].values.astype(bool)

    # Coordinate polari -> cartesiane
    pred_rad = np.deg2rad(pred_angles)
    pred_x = pred_dists * np.cos(pred_rad)
    pred_y = pred_dists * np.sin(pred_rad)

    gt_rad = np.deg2rad(gt_angle_deg)
    gt_x = gt_data[:, 0] * np.cos(gt_rad)
    gt_y = gt_data[:, 0] * np.sin(gt_rad)

    # Metriche solo su frame attivi
    active_mask = gt_active
    if active_mask.any():
        mae_dist = np.mean(np.abs(gt_data[active_mask, 0] - pred_dists[active_mask]))
        diff_rad = np.arctan2(np.sin(pred_rad[active_mask] - gt_rad[active_mask]),
                              np.cos(pred_rad[active_mask] - gt_rad[active_mask]))
        mae_angle = np.mean(np.abs(np.degrees(diff_rad)))
    else:
        mae_dist = mae_angle = float('nan')
    det_acc = np.mean((pred_active == gt_active).astype(float)) * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Spatial Trajectory
    ax = axes[0, 0]
    # GT: grigio quando inattiva, nero quando attiva
    gt_x_inactive = np.where(~gt_active, gt_x, np.nan)
    gt_y_inactive = np.where(~gt_active, gt_y, np.nan)
    gt_x_active   = np.where(gt_active,  gt_x, np.nan)
    gt_y_active   = np.where(gt_active,  gt_y, np.nan)
    ax.plot(gt_x_inactive, gt_y_inactive, color='lightgrey', linewidth=2, label='GT (inactive)')
    ax.plot(gt_x_active,   gt_y_active,   'k--', linewidth=2, alpha=0.8, label='GT (active)')
    # Predizione: solo quando è predetta come attiva
    pred_x_active = np.where(pred_active, pred_x, np.nan)
    pred_y_active = np.where(pred_active, pred_y, np.nan)
    ax.plot(pred_x_active, pred_y_active, 'r-', linewidth=1.5, alpha=0.8, label='Prediction (active)')
    if mic_coords is not None:
        ax.scatter(mic_coords[:, 0], mic_coords[:, 1], c='blue', marker='o', s=80, zorder=5, label='Microphones')
    ax.scatter(gt_x[0], gt_y[0], c='cyan', marker='o', label='Start')
    ax.scatter(gt_x[-1], gt_y[-1], c='black', marker='x', label='End')
    ax.set_title(f'Spatial Trajectory\n(MAE dist={mae_dist:.1f}m, angle={mae_angle:.1f}°)')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 2. Distance over Time
    ax = axes[0, 1]
    gt_dist_inactive = np.where(~gt_active, df['gt_dist'], np.nan)
    gt_dist_active   = np.where(gt_active,  df['gt_dist'], np.nan)
    ax.plot(df['time_s'], gt_dist_inactive, color='lightgrey', linewidth=2, label='GT Dist (inactive)')
    ax.plot(df['time_s'], gt_dist_active,   'k--', linewidth=2, alpha=0.8, label='GT Dist (active)')
    pred_dist_masked = np.where(pred_active, df['pred_dist'], np.nan)
    ax.plot(df['time_s'], pred_dist_masked, 'r-', label='Pred Dist (active)')
    ax.set_ylabel('Distance [m]'); ax.set_title('Distance over Time')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 3. Angle over Time
    ax = axes[1, 1]
    gt_angle_inactive = np.where(~gt_active, df['gt_angle'], np.nan)
    gt_angle_active   = np.where(gt_active,  df['gt_angle'], np.nan)
    ax.plot(df['time_s'], gt_angle_inactive, color='lightgrey', linewidth=2, label='GT Angle (inactive)')
    ax.plot(df['time_s'], gt_angle_active,   'k--', linewidth=2, alpha=0.8, label='GT Angle (active)')
    pred_angle_masked = np.where(pred_active, df['pred_angle'], np.nan)
    ax.plot(df['time_s'], pred_angle_masked, 'r-', label='Pred Angle (active)')
    ax.set_ylabel('Angle [deg]'); ax.set_xlabel('Time [s]')
    ax.set_title('Angle over Time')
    ax.set_ylim(-180, 180); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 4. Siren Detection
    ax = axes[1, 0]
    ax.step(df['time_s'], df['gt_active'], 'k--', where='post', label='GT Active', linewidth=2)
    ax.plot(df['time_s'], df['pred_active_prob'], 'b-', label='Pred Prob', alpha=0.7)
    ax.axhline(0.5, color='orange', linestyle=':', label='Threshold 0.5')
    ax.set_ylabel('Active'); ax.set_xlabel('Time [s]')
    ax.set_title(f'Siren Detection (Acc: {det_acc:.1f}%)')
    ax.set_ylim(-0.05, 1.05); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to: {output_path}")

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
    stats_lines.append(f"Active Frames (GT):  {int(df['gt_active'].sum())} ({df['gt_active'].mean()*100:.1f}%)")
    stats_lines.append("-"*60)

    # Detection stats
    tp = int(((df['gt_active'] == 1) & (df['pred_active'] == 1)).sum())
    fp = int(((df['gt_active'] == 0) & (df['pred_active'] == 1)).sum())
    fn = int(((df['gt_active'] == 1) & (df['pred_active'] == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    det_acc = (df['gt_active'] == df['pred_active']).mean() * 100
    stats_lines.append(f"Detection Accuracy:  {det_acc:.1f}%")
    stats_lines.append(f"Detection Precision: {precision*100:.1f}%")
    stats_lines.append(f"Detection Recall:    {recall*100:.1f}%")
    stats_lines.append(f"Detection F1:        {f1*100:.1f}%")
    stats_lines.append("-"*60)

    # Dist/angle stats only on active frames
    df_active = df[df['gt_active'] == 1]
    if len(df_active) > 0:
        mape_dist = (df_active['error_dist'] / df_active['gt_dist']).mean() * 100
        stats_lines.append(f"Distance MAE:        {df_active['error_dist'].mean():.2f} m  (active frames only)")
        stats_lines.append(f"Distance Median:     {df_active['error_dist'].median():.2f} m")
        stats_lines.append(f"Distance RMSE:       {np.sqrt((df_active['error_dist']**2).mean()):.2f} m")
        stats_lines.append(f"Distance MAPE:       {mape_dist:.1f}%")
        stats_lines.append("-"*60)
        stats_lines.append(f"Angle MAE:           {df_active['error_angle'].mean():.2f}°  (active frames only)")
        stats_lines.append(f"Angle Median:        {df_active['error_angle'].median():.2f}°")
        stats_lines.append(f"Angle RMSE:          {np.sqrt((df_active['error_angle']**2).mean()):.2f}°")
        stats_lines.append(f"Angle Acc <10°:      {(df_active['error_angle'] < 10).mean() * 100:.1f}%")
        stats_lines.append(f"Angle Acc <20°:      {(df_active['error_angle'] < 20).mean() * 100:.1f}%")
    else:
        stats_lines.append("No active frames in this sequence.")
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
    pred_dists, pred_angles, pred_actives, gt_data = inference_on_sequence(model, data, DEVICE)
    
    # 4. Apply Post-Processing (if enabled)
    postprocess_info = {
        'enabled': args.postprocess,
        'method': args.smooth_method if args.postprocess else None,
        'history': args.history if args.postprocess else None
    }
    
    if args.postprocess:
        pred_dists, pred_angles, pred_actives = apply_postprocessing(
            pred_dists, pred_angles, pred_actives,
            method=args.smooth_method, 
            history=args.history
        )
    
    # 5. Create Output Directory
    output_dir = os.path.join(args.output_dir, args.seq)
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Save Predictions CSV
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df = save_predictions_csv(gt_data, pred_dists, pred_angles, pred_actives, csv_path, sample_rate=20)
    
    # 7. Plot Trajectory
    plot_check_path = os.path.join(output_dir, 'trajectory.png')
    plot_trajectory(df, gt_data, pred_dists, pred_angles, plot_check_path, mic_coords=data['microphones'].numpy())
    
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
