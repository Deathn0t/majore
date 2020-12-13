import glob

import kaldiio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AudioFeatureDataset(Dataset):

    def __init__(self, HOW2_PATH, subfolder):

        if subfolder not in ['train', 'test', 'dev5']:
            raise ValueError('subfolder must be either train, val or dev5')
        # Set relative paths for 300h
        self.HOW2_PATH = HOW2_PATH
        self.BASE_PATH = self.HOW2_PATH + 'how2-300h-v1/data/'
        self.AUDIO_PATH = self.HOW2_PATH + 'fbank_pitch_181506/'
        # Read id file
        self.ids = self.get_ids(self.BASE_PATH + f'/{subfolder}/id')

        # Map id-audio
        self.mapping = self.make_dict()

    def get_ids(self, id_file):
        with open(id_file) as f:
            content = f.read()
        return content.strip().split('\n')

    def make_dict(self):
        all_scp = [file for file in glob.glob(self.AUDIO_PATH + '*.scp') if 'raw' in file]
        mapping = dict()

        for scpfile in all_scp:
            with open(scpfile) as f:
                for line in f:
                    video_id, audio_path = line.strip().split()
                    audio_path = audio_path.replace('ARK_PATH/', self.AUDIO_PATH, 1)
                    mapping[video_id] = audio_path
        return mapping

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample_ark = self.mapping[sample_id]
        mat = torch.from_numpy(kaldiio.load_mat(sample_ark))
        padded = torch.zeros((10810, 43))
        padded[:mat.shape[0],: mat.shape[1]] = mat
        return padded


if __name__ == "__main__":

    path_how2 = "/Volumes/LaCie/vision/data/"
    dataset = AudioFeatureDataset(path_how2,"train")
    print("Dataset size: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch in dataloader:
        print(batch.shape)
        break
    pass
