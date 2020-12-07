from torch.utils.data import Dataset
import torch

class MultimodalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, video_dataset, audio_dataset, text_dataset):
        
        self.video_dataset = video_dataset
        self.audio_dataset = audio_dataset
        self.text_dataset = text_dataset

    def __len__(self):
        return len(self.video_dataset)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self.video_dataset[idx]
        audio = self.audio_dataset[idx]
        text  = self.text_dataset[idx]
        
        sample = {"video": video, "audio": audio, "text": text}

        return sample
    
