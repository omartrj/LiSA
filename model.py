import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    """
    Wrapper per applicare la CNN a ogni step temporale della sequenza.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.shape) <= 4:
            return self.module(x)
        b, t = x.shape[0], x.shape[1] 
        # Fonde Batch e Tempo per processarli in parallelo nella CNN
        x_reshape = x.contiguous().view(b * t, *x.shape[2:]) 
        y = self.module(x_reshape)
        # Ripristina la dimensione temporale per la GRU
        y = y.view(b, t, -1)
        return y

class LiSANet(nn.Module):
    def __init__(self, input_channels=8, gru_hidden_size=256, num_gru_layers=2, mic_embedding_dim=16):
        super(LiSANet, self).__init__()
        
        # ENCODER GEOMETRIA MICROFONI
        self.mic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, mic_embedding_dim),
            nn.ReLU()
        )
        
        # CNN BACKBONE
        # Obiettivo: Ridurre la frequenza ma mantenere il Tempo (fondamentale per la fase/TDOA)
        self.cnn_backbone = nn.Sequential(
            # Block 1: 8 -> 32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Pooling Asimmetrico: (2, 1) dimezza Freq, mantiene Tempo
            nn.MaxPool2d(kernel_size=(2, 1)), 
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Global Frequency Pooling
            # Output: (Batch, 256, 1, Time)
            # Media su tutta la frequenza, ma preserva ogni istante temporale
            nn.AdaptiveAvgPool2d((1, None)), 
            nn.Flatten()
        )
        
        # AUTO-CALCOLO DIMENSIONI
        # Crea un input finto per vedere quanto esce dalla CNN
        # Assumiamo input standard: (Batch=1, Channels=8, Freq=128, Time=24)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 128, 24)
            dummy_out = self.cnn_backbone(dummy_input)
            cnn_out_dim = dummy_out.shape[1]
        
        self.time_distributed_cnn = TimeDistributed(self.cnn_backbone)
        
        # GRU
        self.gru = nn.GRU(
            input_size=cnn_out_dim + mic_embedding_dim,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.2 if num_gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # SHARED HEAD
        self.shared_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # HEAD DISTANZA
        self.dist_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        # HEAD ANGOLO (Sin/Cos)
        self.angle_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # HEAD ATTIVITÀ (logit grezzo -> BCEWithLogitsLoss in training, sigmoid in inference)
        self.active_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, mic_coords, hidden_state=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # 1. Mic Geometry Embedding
        mic_embedding = self.mic_encoder(mic_coords)
        
        # 2. CNN Features
        features = self.time_distributed_cnn(x)
        
        # 3. Concatenation
        mic_embedding_expanded = mic_embedding.unsqueeze(1).expand(batch_size, seq_len, -1)
        combined_features = torch.cat([features, mic_embedding_expanded], dim=-1)
        
        # 4. GRU Sequence Modeling
        rnn_out, new_hidden_state = self.gru(combined_features, hidden_state)
        
        # 5. Heads
        shared_features = self.shared_head(rnn_out)
        
        # Distanza (Softplus > 0)
        dist_logit = self.dist_head(shared_features).squeeze(-1)
        dist_pred = torch.nn.functional.softplus(dist_logit)

        # Angolo (Tanh [-1, 1])
        #angle_logits = self.angle_head(shared_features)
        #angle_pred = torch.tanh(angle_logits)
        # BUG: Se uso semplicemente Tanh, è vero che vincolo il seno e il coseno a essere compresi tra -1 e 1, ma non ho alcun vincolo che sen^2(theta) + cos^2(theta) = 1. 
        # La rete allora prova a  minimizzare la loss semplicemente mettendo sin e cos vicini a zero, invece di imparare la relazione trigonometrica.
        # FIX (???): normalizzazione L2 del vettore (sin, cos) in uscita. 
        # In questo modo la rete è costretta a scegliere un punto nel cerchio unitario, garantendo che sen^2 + cos^2 = 1.
        angle_logits = self.angle_head(shared_features)
        angle_pred = torch.nn.functional.normalize(angle_logits, p=2, dim=-1, eps=1e-6)
        
        # Attività: logit grezzo (shape: B, SeqLen)
        # Usare BCEWithLogitsLoss durante il training per stabilità numerica
        active_logit = self.active_head(shared_features).squeeze(-1)
        
        return dist_pred, angle_pred, active_logit, new_hidden_state

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2)
    total_params = count_parameters(model)
    print(f"LiSANet Total Trainable Parameters: {total_params}")

if __name__ == "__main__":
    main()