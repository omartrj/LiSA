import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
import json
from collections import OrderedDict

class LiSADataset(Dataset):
    def __init__(self, file_list, root_dir='data/processed', seq_len=50, stride=25, max_cache_size=150):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride
        self.index_map = []
        
        # LRU Cache: 150 files * 150MB approx 22GB RAM
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()
        
        # Metadata cache to avoid loading 150GB at startup
        meta_path = os.path.join(root_dir, "metadata_cache.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        updated_meta = False
        for seq_file in file_list:
            if seq_file in metadata:
                num_samples = metadata[seq_file]
            else:
                # Load only if metadata is missing
                seq_path = os.path.join(self.root_dir, seq_file)
                data = torch.load(seq_path, weights_only=True)
                num_samples = data['gt'].shape[0]
                metadata[seq_file] = num_samples
                updated_meta = True
            
            for start_idx in range(0, num_samples - seq_len + 1, stride):
                self.index_map.append((seq_file, start_idx))
        
        if updated_meta:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
        
        # Keep files grouped to maximize cache hits
        self.index_map.sort(key=lambda x: x[0])
        print(f"Dataset indexed: {len(self.index_map)} sequences. Cache: {max_cache_size} files.")

    def __len__(self):
        return len(self.index_map)
    
    def _load_file(self, filename):
        if filename in self.cache:
            self.cache.move_to_end(filename)
            return self.cache[filename]
        
        seq_path = os.path.join(self.root_dir, filename)
        data = torch.load(seq_path, weights_only=False)
        
        self.cache[filename] = data
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        filename, start_idx = self.index_map[idx]
        end_idx = start_idx + self.seq_len
        
        data = self._load_file(filename)
        
        spec_seq = data['spectrograms'][start_idx:end_idx]
        gt_seq = data['gt'][start_idx:end_idx]
        mic_coords = data['microphones']
        
        # Optimized tensor slicing and stacking
        # gt_seq contains (dist, sin, cos, is_active)
        return {
            'spectrogram': spec_seq,
            'gt_dist': gt_seq[:, 0].float(),
            'gt_angle': gt_seq[:, 1:3].float(),  # Returns (Seq, 2) [sin, cos]
            'gt_active': gt_seq[:, 3].float(),    # Returns (Seq,) [0 or 1]
            'microphones': mic_coords.float()
        }

def get_dataloaders(batch_size=32, val_split=0.2, test_split=0.1, 
                    processed_dir='data/processed', seq_len=50, seed=420):
    all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    random.seed(seed)
    random.shuffle(all_files)
    
    total_files = len(all_files)
    test_count = int(total_files * test_split)
    val_count = int(total_files * val_split)
    train_count = total_files - test_count - val_count
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count+val_count]
    test_files = all_files[train_count+val_count:]
    
    # Large cache for training, smaller for validation
    train_dataset = LiSADataset(train_files, processed_dir, seq_len, stride=seq_len//2, max_cache_size=150)
    val_dataset = LiSADataset(val_files, processed_dir, seq_len, stride=seq_len, max_cache_size=50)
    test_dataset = LiSADataset(test_files, processed_dir, seq_len, stride=seq_len, max_cache_size=50)
    
    # shuffle=False is mandatory to maintain file-sequential access for cache efficiency
    loader_args = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True, 'shuffle': False}
    
    return (DataLoader(train_dataset, **loader_args),
            DataLoader(val_dataset, **loader_args),
            DataLoader(test_dataset, **loader_args))

def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, seq_len=50)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()