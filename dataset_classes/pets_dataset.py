from torch.utils.data import Dataset
from typing import List
from pathlib import Path
from PIL import Image

class PetsDataset(Dataset):
    def __init__(self, filenames:List[Path], transform):
        self.filenames = filenames
        self.transform = transform
    def __getitem__(self, idx):
        path = self.filenames[idx]
        if not isinstance(path, list):
            path = [path]
        imgs = []
        labels = []
        for p in path:
            img = Image.open(p).convert('RGB')
            img = self.transform(img)
            # it's a cat picture if the first letter of the file name is capital
            label = 1 if p.name[0].isupper() else 0
            imgs.append(img)
            labels.append(label)
            
        return imgs, labels
        
    def __len__(self):
        return len(self.filenames)