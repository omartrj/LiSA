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
    def __init__(self, alpha=1.0, beta=1.0):
        """Loss personalizzata che combina errore di distanza e angolo.

        alpha: peso per la componente di distanza
        beta: peso per la componente di angolo
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_dist, target_dist, pred_angle, target_angle):
        # 1. LOSS DISTANZA (Log-MSE)
        # Questo penalizza gli errori relativi invece di quelli assoluti
        log_pred = torch.log1p(pred_dist)
        log_target = torch.log1p(target_dist)
        loss_dist = self.mse(log_pred, log_target)
        
        # 2. LOSS ANGOLO (MSE su sin e cos)
        pred_sin = pred_angle[:, :, 0]
        pred_cos = pred_angle[:, :, 1]
        target_sin = target_angle[:, :, 0]
        target_cos = target_angle[:, :, 1]
        
        loss_sin = self.mse(pred_sin, target_sin)
        loss_cos = self.mse(pred_cos, target_cos)
        loss_angle = loss_sin + loss_cos
        
        total_loss = self.alpha * loss_dist + self.beta * loss_angle
        
        return total_loss, self.alpha * loss_dist, self.beta * loss_angle
    
def compute_metrics(pred_dist, target_dist, pred_angle, target_angle):
    """
    Calcola metriche interpretabili per l'uomo (Metri e Gradi).
    """
    # 1. Errore Distanza (Mean Absolute Error)
    mae_dist = torch.mean(torch.abs(pred_dist - target_dist)).item()
    
    # 2. Errore Angolo (Gradi)
    # Estrai sin e cos
    pred_sin = pred_angle[:, :, 0]
    pred_cos = pred_angle[:, :, 1]
    target_sin = target_angle[:, :, 0]
    target_cos = target_angle[:, :, 1]
    
    # Converti sin/cos in angoli (radianti)
    pred_angle_rad = torch.atan2(pred_sin, pred_cos)
    target_angle_rad = torch.atan2(target_sin, target_cos)
    
    # Differenza angolare
    diff_rad = pred_angle_rad - target_angle_rad
    diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))  # Normalizza in [-π, π]
    
    mae_angle = torch.mean(torch.abs(torch.rad2deg(diff_rad))).item()
    
    return mae_dist, mae_angle

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_loss_dist = 0.0
    running_loss_angle = 0.0

    running_dist_mae = 0.0
    running_angle_mae = 0.0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        specs = batch['spectrogram'].to(DEVICE)       # (B, Seq, 8, F, T)
        gt_dist = batch['gt_dist'].to(DEVICE)         # (B, Seq)
        gt_angle = batch['gt_angle'].to(DEVICE)       # (B, Seq, 2) [sin, cos]
        mic_coords = batch['microphones'].to(DEVICE)  # (B, 4, 3)
        
        optimizer.zero_grad()
        
        # Forward Pass
        pred_dist, pred_angle, _ = model(specs, mic_coords, hidden_state=None)
        
        # Calcolo Loss
        loss, loss_dist, loss_angle = criterion(pred_dist, gt_dist, pred_angle, gt_angle)
        
        # Backward Pass
        loss.backward()
        
        # Gradient Clipping per la GRU
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log
        running_loss += loss.item()
        running_loss_dist += loss_dist.item()
        running_loss_angle += loss_angle.item()

        mae_d, mae_a = compute_metrics(pred_dist, gt_dist, pred_angle, gt_angle)
        running_dist_mae += mae_d
        running_angle_mae += mae_a
        
        pbar.set_postfix({'L': f"{loss.item():.4f}", 'Dist MAE': f"{mae_d:.2f}m", 'Angle MAE': f"{mae_a:.2f}°"})
        
    n_batches = len(loader)
    return running_loss / n_batches, running_loss_dist / n_batches, running_loss_angle / n_batches, running_dist_mae / n_batches, running_angle_mae / n_batches

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_loss_dist = 0.0
    running_loss_angle = 0.0

    running_dist_mae = 0.0
    running_angle_mae = 0.0
    
    with torch.no_grad():
        for batch in loader:
            specs = batch['spectrogram'].to(DEVICE)
            gt_dist = batch['gt_dist'].to(DEVICE)
            gt_angle = batch['gt_angle'].to(DEVICE)
            mic_coords = batch['microphones'].to(DEVICE)
            
            # Forward
            pred_dist, pred_angle, _ = model(specs, mic_coords, hidden_state=None)
            
            # Loss
            loss, loss_dist, loss_angle = criterion(pred_dist, gt_dist, pred_angle, gt_angle)
            
            # Metrics
            mae_d, mae_a = compute_metrics(pred_dist, gt_dist, pred_angle, gt_angle)
            
            running_loss += loss.item()
            running_loss_dist += loss_dist.item()
            running_loss_angle += loss_angle.item()
            
            running_dist_mae += mae_d
            running_angle_mae += mae_a
            
    n_batches = len(loader)
    return running_loss / n_batches, running_loss_dist / n_batches, running_loss_angle / n_batches, running_dist_mae / n_batches, running_angle_mae / n_batches

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
    criterion = MultiLoss(alpha=1.0, beta=1.0)
    
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
        t_loss, t_loss_dist, t_loss_angle, t_dist_mae, t_angle_mae = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"TRAIN | Loss: {t_loss:.4f} ({t_loss_angle:.4f}[ld] + {t_loss_dist:.4f}[la]) | Dist Err: {t_dist_mae:.2f}m | Angle Err: {t_angle_mae:.2f}°")
        
        # Val
        v_loss, v_loss_dist, v_loss_angle, v_dist_mae, v_angle_mae = validate(model, val_loader, criterion)
        print(f"VAL   | Loss: {v_loss:.4f} ({v_loss_angle:.4f}[ld] + {v_loss_dist:.4f}[la]) | Dist Err: {v_dist_mae:.2f}m | Angle Err: {v_angle_mae:.2f}°")
        
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