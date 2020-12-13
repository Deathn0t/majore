import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

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

