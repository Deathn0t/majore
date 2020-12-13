import os

import numpy as np
from kaldiio import ReadHelper
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AudioFeatureDataset(Dataset):

    def __init__(self, how2_path, subfolder):

        if subfolder not in ['train', 'test', 'dev5']:
            raise ValueError('subfolder must be either train, val or dev5')
        # Set relative paths for 300h
        self.how2_path= how2_path
        self.base_path= os.path.join(self.how2_path, "how2-300h-v1/data/", subfolder)
        # Read id file
        self.id_path = os.path.join(self.base_path, "id")
        self.scp_path = os.path.join(self.base_path, "feats.scp")
        self.videoId2index = self.compute_ids()
        self.data = np.zeros((len(self), 10810, 43), dtype=np.float32)
        self.load_kaldi()

    def compute_ids(self):
        with open(self.id_path, "r") as f:
            mapping = {videoId.split("\n")[0]:i for i, videoId in enumerate(f)}
        return mapping

    def load_kaldi(self):
        with ReadHelper(f"scp:{self.scp_path}") as reader:
            for key, mat in tqdm(reader):
                i = self.videoId2index[key]
                self.data[i, :mat.shape[0],: mat.shape[1]] = mat[:,:]

    def __len__(self):
        return len(self.videoId2index)

    def __getitem__(self, idx):
        return self.data[idx]