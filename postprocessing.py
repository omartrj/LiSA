import numpy as np
from collections import deque

class PostProcessor:
    def __init__(self, history_length=5, method='median'):
        """
        Gestisce lo smoothing temporale delle predizioni.
        
        Args:
            history_length (int): Quanti frame passati considerare (es. 5 frame = 0.5s circa).
            method (str): 'mean' (Media Mobile) o 'median' (Filtro Mediano).
        """
        self.history_length = history_length
        self.method = method
        
        # Buffer per distanze e angoli (in gradi)
        self.dist_buffer = deque(maxlen=history_length)
        self.angle_buffer = deque(maxlen=history_length)
        
    def reset(self):
        """Pulisce la memoria (utile quando cambia la sequenza)."""
        self.dist_buffer.clear()
        self.angle_buffer.clear()
        
    def update(self, raw_dist, raw_angle):
        """
        Aggiunge una nuova predizione grezza e restituisce quella filtrata.
        Usa np.unwrap per gestire correttamente il wrap-around degli angoli (es. -179° e 179°).
        
        Args:
            raw_dist (float): Distanza predetta grezza.
            raw_angle (float): Angolo predetto grezzo in gradi.
            
        Returns:
            (float, float): Distanza e angolo filtrati.
        """
        # Aggiungi ai buffer
        self.dist_buffer.append(raw_dist)
        self.angle_buffer.append(raw_angle)
        
        # Se non ho abbastanza dati, restituisco l'input grezzo
        if len(self.angle_buffer) < 1:
            return raw_dist, raw_angle

        # Conversione deque -> numpy array per calcoli
        dists = np.array(self.dist_buffer)
        angles = np.array(self.angle_buffer)
        
        # Converti angoli in radianti per usare np.unwrap
        angles_rad = np.deg2rad(angles)
        
        # Applica unwrap per gestire i salti di ±360°
        angles_rad_unwrapped = np.unwrap(angles_rad)
        
        if self.method == 'median':
            # Filtro Mediano (Robustissimo contro outliers/picchi)
            smooth_dist = np.median(dists)
            smooth_angle_rad = np.median(angles_rad_unwrapped)
        else:
            # Media Mobile (Più fluido, ma soffre i picchi)
            smooth_dist = np.mean(dists)
            smooth_angle_rad = np.mean(angles_rad_unwrapped)
            
        # Riconverti in gradi
        smooth_angle = np.rad2deg(smooth_angle_rad)
        
        # Normalizza tra -180 e 180
        smooth_angle = ((smooth_angle + 180) % 360) - 180
            
        return smooth_dist, smooth_angle