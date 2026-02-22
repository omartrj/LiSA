import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm

from dataset import get_dataloaders
from model import LiSANet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
DATA_DIR = 'data'

class MultiLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0):
        """Loss personalizzata che combina errore di distanza, angolo e rilevamento sirena.

        alpha: peso per la componente di distanza
        beta:  peso per la componente di angolo
        gamma: peso per la componente di attività (BCE)
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred_dist, target_dist, pred_angle, target_angle,
                pred_active_logit, target_active):
        # Maschera: distanza e angolo si calcolano SOLO quando la sirena è attiva.
        # Così la rete non viene penalizzata per predizioni meaningless durante il silenzio.
        mask = target_active.bool()  # (B, Seq)
        
        # 1. LOSS DISTANZA (Log-MSE, solo su frame attivi)
        log_pred = torch.log1p(pred_dist)
        log_target = torch.log1p(target_dist)
        loss_dist_raw = self.mse(log_pred, log_target)  # (B, Seq)
        if mask.any():
            loss_dist = loss_dist_raw[mask].mean()
        else:
            loss_dist = loss_dist_raw.mean() * 0.0  # zero gradient-friendly
        
        # 2. LOSS ANGOLO (MSE su sin e cos, solo su frame attivi)
        pred_sin   = pred_angle[:, :, 0]
        pred_cos   = pred_angle[:, :, 1]
        target_sin = target_angle[:, :, 0]
        target_cos = target_angle[:, :, 1]
        
        loss_sin_raw = self.mse(pred_sin, target_sin)  # (B, Seq)
        loss_cos_raw = self.mse(pred_cos, target_cos)  # (B, Seq)
        if mask.any():
            loss_angle = loss_sin_raw[mask].mean() + loss_cos_raw[mask].mean()
        else:
            loss_angle = (loss_sin_raw + loss_cos_raw).mean() * 0.0
        
        # 3. LOSS ATTIVITÀ (BCE, su tutti i frame)
        loss_active = self.bce(pred_active_logit, target_active)
        
        total_loss = self.alpha * loss_dist + self.beta * loss_angle + self.gamma * loss_active
        
        return total_loss, self.alpha * loss_dist, self.beta * loss_angle, self.gamma * loss_active
    
def compute_metrics(pred_dist, target_dist, pred_angle, target_angle, pred_active_logit, target_active):
    """
    Calcola metriche interpretabili per l'uomo (Metri, Gradi, Accuracy rilevamento).
    Dist e Angle vengono calcolati solo sui frame in cui la sirena è attiva.
    """
    mask = target_active.bool()  # (B, Seq)

    # 1. Errore Distanza (MAE, solo su frame attivi)
    if mask.any():
        mae_dist = torch.mean(torch.abs(pred_dist[mask] - target_dist[mask])).item()
    else:
        mae_dist = float('nan')
    
    # 2. Errore Angolo (Gradi, solo su frame attivi)
    pred_sin   = pred_angle[:, :, 0]
    pred_cos   = pred_angle[:, :, 1]
    target_sin = target_angle[:, :, 0]
    target_cos = target_angle[:, :, 1]
    
    pred_angle_rad   = torch.atan2(pred_sin, pred_cos)
    target_angle_rad = torch.atan2(target_sin, target_cos)
    diff_rad = torch.atan2(torch.sin(pred_angle_rad - target_angle_rad),
                           torch.cos(pred_angle_rad - target_angle_rad))
    if mask.any():
        mae_angle = torch.mean(torch.abs(torch.rad2deg(diff_rad[mask]))).item()
    else:
        mae_angle = float('nan')
    
    # 3. Accuracy rilevamento sirena
    pred_active = (torch.sigmoid(pred_active_logit) >= 0.5)
    acc_active = (pred_active == mask).float().mean().item()
    
    return mae_dist, mae_angle, acc_active

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_loss_dist = 0.0
    running_loss_angle = 0.0
    running_dist_mae = 0.0
    running_angle_mae = 0.0
    
    running_loss_active = 0.0
    running_active_acc = 0.0

    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        specs      = batch['spectrogram'].to(DEVICE)  # (B, Seq, 8, F, T)
        gt_dist    = batch['gt_dist'].to(DEVICE)      # (B, Seq)
        gt_angle   = batch['gt_angle'].to(DEVICE)     # (B, Seq, 2) [sin, cos]
        gt_active  = batch['gt_active'].to(DEVICE)    # (B, Seq)
        mic_coords = batch['microphones'].to(DEVICE)  # (B, 4, 3)
        
        optimizer.zero_grad()
        
        # Forward Pass
        pred_dist, pred_angle, pred_active_logit, _ = model(specs, mic_coords, hidden_state=None)
        
        # Calcolo Loss
        loss, loss_dist, loss_angle, loss_active = criterion(
            pred_dist, gt_dist, pred_angle, gt_angle, pred_active_logit, gt_active
        )
        
        # Backward Pass
        loss.backward()
        
        # Gradient Clipping per la GRU
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log
        running_loss += loss.item()
        running_loss_dist += loss_dist.item()
        running_loss_angle += loss_angle.item()
        running_loss_active += loss_active.item()

        mae_d, mae_a, acc = compute_metrics(pred_dist, gt_dist, pred_angle, gt_angle, pred_active_logit, gt_active)
        if not (mae_d != mae_d):  # skip nan
            running_dist_mae += mae_d
        if not (mae_a != mae_a):
            running_angle_mae += mae_a
        running_active_acc += acc
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
    n_batches = len(loader)
    return (running_loss / n_batches, running_loss_dist / n_batches,
            running_loss_angle / n_batches, running_loss_active / n_batches,
            running_dist_mae / n_batches, running_angle_mae / n_batches,
            running_active_acc / n_batches)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_loss_dist = 0.0
    running_loss_angle = 0.0
    running_loss_active = 0.0
    running_dist_mae = 0.0
    running_angle_mae = 0.0
    running_active_acc = 0.0
    
    with torch.no_grad():
        for batch in loader:
            specs      = batch['spectrogram'].to(DEVICE)
            gt_dist    = batch['gt_dist'].to(DEVICE)
            gt_angle   = batch['gt_angle'].to(DEVICE)
            gt_active  = batch['gt_active'].to(DEVICE)
            mic_coords = batch['microphones'].to(DEVICE)
            
            # Forward
            pred_dist, pred_angle, pred_active_logit, _ = model(specs, mic_coords, hidden_state=None)
            
            # Loss
            loss, loss_dist, loss_angle, loss_active = criterion(
                pred_dist, gt_dist, pred_angle, gt_angle, pred_active_logit, gt_active
            )
            
            # Metrics
            mae_d, mae_a, acc = compute_metrics(
                pred_dist, gt_dist, pred_angle, gt_angle, pred_active_logit, gt_active
            )
            
            running_loss += loss.item()
            running_loss_dist += loss_dist.item()
            running_loss_angle += loss_angle.item()
            running_loss_active += loss_active.item()
            if not (mae_d != mae_d):
                running_dist_mae += mae_d
            if not (mae_a != mae_a):
                running_angle_mae += mae_a
            running_active_acc += acc
            
    n_batches = len(loader)
    return (running_loss / n_batches, running_loss_dist / n_batches,
            running_loss_angle / n_batches, running_loss_active / n_batches,
            running_dist_mae / n_batches, running_angle_mae / n_batches,
            running_active_acc / n_batches)

def main(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Starting Training with the following configuration:")
    print(f"Device: {DEVICE}")
    print(f"Seq Length: {args.seq_len}")
    print(f"Batch Size: {args.batch_size}")
    
    # Dataset
    print("Loading Datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        processed_dir=f"{DATA_DIR}/processed",
        seq_len=args.seq_len
    )
    
    # Model
    print("Initializing Model...")
    model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = MultiLoss(alpha=1.0, beta=1.0, gamma=2.0)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Paths
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    last_model_path = os.path.join(CHECKPOINT_DIR, "last_model.pth")
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(last_model_path):
        print(f"Resuming training from checkpoint: {last_model_path}")
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Starting Training Loop...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        t_loss, t_loss_dist, t_loss_angle, t_loss_active, t_dist_mae, t_angle_mae, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"TRAIN | Loss: {t_loss:.4f} (dist={t_loss_dist:.4f} angle={t_loss_angle:.4f} active={t_loss_active:.4f}) | Dist MAE: {t_dist_mae:.2f}m | Angle MAE: {t_angle_mae:.2f}° | Det Acc: {t_acc*100:.1f}%")
        
        # Val
        v_loss, v_loss_dist, v_loss_angle, v_loss_active, v_dist_mae, v_angle_mae, v_acc = validate(model, val_loader, criterion)
        print(f"VAL   | Loss: {v_loss:.4f} (dist={v_loss_dist:.4f} angle={v_loss_angle:.4f} active={v_loss_active:.4f}) | Dist MAE: {v_dist_mae:.2f}m | Angle MAE: {v_angle_mae:.2f}° | Det Acc: {v_acc*100:.1f}%")
        
        # Scheduling
        scheduler.step(v_loss)
        
        # Checkpointing
        if v_loss < best_val_loss:
            print(f"--> New Best Model! (Loss: {best_val_loss:.4f} -> {v_loss:.4f})")
            best_val_loss = v_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            print(f"No improvement. Early Stop Counter: {early_stop_counter+1}/{args.patience}")
            early_stop_counter += 1
            
        # Save Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': v_loss,
        }, last_model_path)
        
        # Early Stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs of no improvement.")
            break
            
    print("\nTraining Finished.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50, help="Lunghezza sequenza temporale (es. 50 step = 2.5s)")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=11, help="Epochs di pazienza per Early Stopping")
    parser.add_argument("--resume", action='store_true', help="Riprendi l'allenamento dal checkpoint più recente")
    
    args = parser.parse_args()
    main(args)