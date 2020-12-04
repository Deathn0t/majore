import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


path_valid = "how2-300h-v1/data/val"

class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_texts):
        self.path_texts = path_texts # avg pool
        self.texts = self.load_texts()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts = self.texts[idx]
        sample = {"text": texts}

        return sample

    def load_texts(self):
        
        with open(self.path_texts, "r") as f:
            lines = f.readlines()
            
        clean = []
        for line in lines:
            clean.append(line.strip('\n'))
            
        return  np.array(clean)


if __name__ == "__main__":

    path_how2 = "/Volumes/LaCie/vision/data/"
    path_texts = os.path.join(path_how2, "how2-300h-v1/data/train", "text.en")
    dataset = TextDataset(path_texts)
    print("Dataset size: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch in dataloader:
        print(type(batch["text"]))
        print(len(batch["text"]))
        print(batch["text"])
        
        break
