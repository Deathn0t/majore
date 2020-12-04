import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


path_valid = "how2-300h-v1/data/val"

class VideoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_videos):
        self.path_videos = path_videos # avg pool
        self.videos = self.load_videos()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        videos = self.videos[idx, :]

        sample = {"video": videos}

        return sample

    def load_videos(self):
        with open(self.path_videos, "rb") as fd:
            videos = np.load(fd)
        return videos



if __name__ == "__main__":
    path_how2 = "/Volumes/T7/University/Polytechnique/INF634-Advanced-Computer-Vision/data"
    path_videos = os.path.join(path_how2, "resnext101-action-avgpool-300h", "train.npy")
    dataset = VideoDataset(path_videos)

    print("Dataset size: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch in dataloader:
        print(type(batch["video"]))
        print(batch["video"].size())
        break

