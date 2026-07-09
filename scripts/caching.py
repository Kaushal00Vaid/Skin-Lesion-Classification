import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms

split = pd.read_csv("../data/HAM10000_metadata_split_80_20.csv")
train_df = split[split['split'] == "train"]
val_df = split[split['split'] == "val"]

IMG_SIZE = 380 # EffNet-B4 Native

img_dirs = [
    '../data/HAM10000_images_part_1',
    '../data/HAM10000_images_part_2'
]
label_map = {label : idx for idx, label in enumerate(sorted(split['dx'].unique()))}

path_map = {}
for d in img_dirs:
    for fname in os.listdir(d):
        if fname.endswith('.jpg'):
            image_id = fname[:-4] # strip '.jpg'
            path_map[image_id] = os.path.join(d, fname)

resize_only = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()  # keep as tensor, normalize later at __getitem__ (cheap, no I/O)
])

def build_cache(df, path_map, save_path):
    images = torch.zeros((len(df), 3, 380, 380), dtype=torch.float32)
    for i, row in df.reset_index(drop=True).iterrows():
        img = Image.open(path_map[row['image_id']]).convert('RGB')
        images[i] = resize_only(img)
        if i % 1000 == 0:
            print(f"{i}/{len(df)}")
    torch.save(images, save_path)
    print(f"Saved cache to {save_path}")

# build once, reuse across all 20 epochs
build_cache(train_df, path_map, '../data/train_cache.pt')
build_cache(val_df, path_map, '../data/val_cache.pt')