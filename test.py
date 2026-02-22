import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataset import get_dataloaders
from model import LiSANet
from postprocessing import PostProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_errors(pred_dist, target_dist, pred_angle_deg, target_angle_deg):
    err_dist_abs = np.abs(pred_dist - target_dist)
    err_dist_pct = (err_dist_abs / (target_dist + 1e-6)) * 100
    
    diff = pred_angle_deg - target_angle_deg
    diff = (diff + 180) % 360 - 180
    err_angle = np.abs(diff)
    
    return err_dist_abs, err_dist_pct, err_angle

def plot_error_by_distance(target_dist, pred_dist, target_angle, pred_angle, output_dir):
    err_dist = np.abs(pred_dist - target_dist)
    diff_angle = (pred_angle - target_angle + 180) % 360 - 180
    err_angle = np.abs(diff_angle)
    
    bins = np.arange(0, np.max(target_dist) + 5, 5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    dist_mae = []
    angle_mae = []
    
    for i in range(len(bins)-1):
        mask = (target_dist >= bins[i]) & (target_dist < bins[i+1])
        if np.any(mask):
            dist_mae.append(np.mean(err_dist[mask]))
            angle_mae.append(np.mean(err_angle[mask]))
        else:
            dist_mae.append(np.nan)
            angle_mae.append(np.nan)
            
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bin_centers, dist_mae, marker='o', linestyle='-', color='b')
    plt.xlabel("True Distance (m)")
    plt.ylabel("Distance MAE (m)")
    plt.title("Distance Error vs True Distance")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, angle_mae, marker='o', linestyle='-', color='r')
    plt.xlabel("True Distance (m)")
    plt.ylabel("Angle MAE (°)")
    plt.title("Angle Error vs True Distance")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_distance.png"))
    plt.close()

def evaluate_model(model, loader):
    model.eval()
    all_pred_dist, all_target_dist = [], []
    all_pred_angle, all_target_angle = [], []
    all_pred_active, all_target_active = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            specs      = batch['spectrogram'].to(DEVICE)
            gt_dist    = batch['gt_dist'].to(DEVICE)
            gt_angle   = batch['gt_angle'].to(DEVICE)   # (B, Seq, 2) [sin, cos]
            gt_active  = batch['gt_active'].to(DEVICE)  # (B, Seq)
            mic_coords = batch['microphones'].to(DEVICE)
            
            # Forward: nuovo output include active_logit
            pred_dist, pred_angle, pred_active_logit, _ = model(specs, mic_coords, hidden_state=None)
            
            # Estrai sin e cos per conversione in gradi
            pred_sin = pred_angle[:, :, 0]
            pred_cos = pred_angle[:, :, 1]
            gt_sin   = gt_angle[:, :, 0]
            gt_cos   = gt_angle[:, :, 1]
            
            # Converti predizioni sin/cos in angoli (gradi)
            pred_angle_rad = torch.atan2(pred_sin, pred_cos)
            pred_angle_deg = torch.rad2deg(pred_angle_rad)
            
            # Converti target sin/cos in angoli (gradi)
            target_angle_rad = torch.atan2(gt_sin, gt_cos)
            target_angle_deg = torch.rad2deg(target_angle_rad)
            
            # Probabilità attività (sigmoid del logit)
            pred_active_prob = torch.sigmoid(pred_active_logit)
            
            all_pred_dist.extend(pred_dist.cpu().numpy().flatten())
            all_target_dist.extend(gt_dist.cpu().numpy().flatten())
            all_pred_angle.extend(pred_angle_deg.cpu().numpy().flatten())
            all_target_angle.extend(target_angle_deg.cpu().numpy().flatten())
            all_pred_active.extend(pred_active_prob.cpu().numpy().flatten())
            all_target_active.extend(gt_active.cpu().numpy().flatten())
            
    return (np.array(all_pred_dist), np.array(all_target_dist),
            np.array(all_pred_angle), np.array(all_target_angle),
            np.array(all_pred_active), np.array(all_target_active))

def main(args):
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, seq_len=args.seq_len, processed_dir='data/processed')
    
    model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
    p_dist, t_dist, p_angle, t_angle, p_active_prob, t_active = evaluate_model(model, test_loader)
    
    if args.postprocess:
        processor = PostProcessor(history_length=args.history, method=args.smooth_method)
        p_dist_smooth, p_angle_smooth = [], []
        for d, a in zip(p_dist, p_angle):
            sd, sa = processor.update(d, a)
            p_dist_smooth.append(sd)
            p_angle_smooth.append(sa)
        p_dist, p_angle = np.array(p_dist_smooth), np.array(p_angle_smooth)
    
    # Maschera frame attivi per dist/angle
    active_mask = t_active.astype(bool)
    p_active_bin = (p_active_prob >= 0.5).astype(int)
    t_active_int = t_active.astype(int)

    # Detection metrics
    tp = int(((t_active_int == 1) & (p_active_bin == 1)).sum())
    fp = int(((t_active_int == 0) & (p_active_bin == 1)).sum())
    fn = int(((t_active_int == 1) & (p_active_bin == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    det_acc = float((t_active_int == p_active_bin).mean())

    # Dist/angle errors (active frames only)
    if active_mask.any():
        err_dist_abs, err_dist_pct, err_angle = compute_errors(
            p_dist[active_mask], t_dist[active_mask],
            p_angle[active_mask], t_angle[active_mask]
        )
    else:
        err_dist_abs = err_dist_pct = err_angle = np.array([])
    
    metrics = {
        "count": int(len(p_dist)),
        "active_frames": int(active_mask.sum()),
        "postprocessed": {
            "enabled": args.postprocess,
            "method": args.smooth_method if args.postprocess else "none",
            "history_length": args.history if args.postprocess else 0
        },
        "detection": {
            "accuracy": det_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "angle": {
            "mae":    float(np.mean(err_angle))    if len(err_angle) else float('nan'),
            "median": float(np.median(err_angle))  if len(err_angle) else float('nan'),
            "rmse":   float(np.sqrt(np.mean(err_angle**2))) if len(err_angle) else float('nan')
        },
        "distance": {
            "mae":    float(np.mean(err_dist_abs))  if len(err_dist_abs) else float('nan'),
            "median": float(np.median(err_dist_abs)) if len(err_dist_abs) else float('nan'),
            "rmse":   float(np.sqrt(np.mean(err_dist_abs**2))) if len(err_dist_abs) else float('nan'),
            "mape":   float(np.mean(err_dist_pct)) if len(err_dist_pct) else float('nan')
        }
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Detection Acc:  {det_acc*100:.1f}%  |  Precision: {precision*100:.1f}%  |  Recall: {recall*100:.1f}%  |  F1: {f1*100:.1f}%")
    print(f"Angle MAE:      {metrics['angle']['mae']:.2f}°  (active frames only)")
    print(f"Dist MAE:       {metrics['distance']['mae']:.2f} m  (active frames only)")
        
    if active_mask.any():
        plot_error_by_distance(t_dist[active_mask], p_dist[active_mask],
                               t_angle[active_mask], p_angle[active_mask], args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="test_results")

    # Post-processing options
    parser.add_argument("--postprocess", action='store_true')
    parser.add_argument("--smooth_method", type=str, default="median")
    parser.add_argument("--history", type=int, default=5)
    args = parser.parse_args()
    main(args)