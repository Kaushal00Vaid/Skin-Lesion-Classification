
import torch
from torch.utils.data import Dataset, DataLoader

from caching import label_map, train_df, val_df

class CachedHAMDataset(Dataset):
    def __init__(self, cache_path, df, label_map):
        self.images =torch.load(cache_path)
        self.labels = [label_map[l] for l in df['dx']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_cache_path = "../data/train_cache.pt"
val_cache_path = "../data/val_cache.pt"

train_dataset = CachedHAMDataset(train_cache_path, train_df, label_map)
val_dataset = CachedHAMDataset(val_cache_path, val_df, label_map)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
