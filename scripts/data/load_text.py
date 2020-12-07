import os
import re
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_texts, path_embeddings):
        
        self.path_texts = path_texts # avg pool
        self.texts, self.splitted, self.max_seg_word = self.load_texts()
        self.mapping = self.load_embeddings(path_embeddings)
        self.size_emb = len(list(self.mapping["I"]))
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.texts[idx]
        splitted = self.splitted[idx]
        
        embeddings = self.extract_embedding(splitted)
        
        sample = {"text": text, "embedding": embeddings}

        return sample
    
    def extract_embedding(self, splitted):
        
        embeddings = np.zeros((self.max_seg_word, self.size_emb), dtype = float)
    
        index = 0
        
        for word in splitted:
            
            if len(word)>=1:
                
                word_upper = word.upper()
                
                if word_upper in self.mapping:
                    
                    emb = list(self.mapping[word_upper])
                    embeddings[index,:] = emb
                    index+=1
         
        return embeddings        
                            
                 
    def load_texts(self):
        
        with open(self.path_texts, "r") as f:
            lines = f.readlines()
            
        text = []
        splitted = []
        
        largest_split = 0
        
        for line in lines:
            
            strip = line.strip('\n')
            text.append(strip)
            
            split = re.sub(r"[^\w\d'\s]+",'',strip).split(" ")
            splitted.append(split)
            
            count_split = len(split)
            
            if count_split > largest_split:
                largest_split = count_split
                
            splitted.append(split)
            
        print(len(lines))
            
        return  np.array(text), np.array(splitted), largest_split
    
    def load_embeddings(self,fname):
        
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        
        dict_words = dict()
        for k,v in data.items():
            dict_words[k] = list(v)
        return dict_words
    
if  __name__ == "__main__":

    path_how2 = "/Volumes/LaCie/vision/data/"
    
    path_texts = os.path.join(path_how2, "how2-300h-v1/data/train", "text.en")
    
    path_embeddings = os.path.join(path_how2, "how2-release/word_embedding/","cmu_partition.train.vec")
    
    dataset = TextDataset(path_texts, path_embeddings)
    print("Dataset size: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch in dataloader:
    
        print(type(batch["text"]))
        print(len(batch["text"]))
        print(batch["text"])
        print(batch["embedding"])
        print(batch["embedding"].shape)
        
        break
