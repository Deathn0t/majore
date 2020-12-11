import os
import re
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):

    def __init__(self, path_texts, path_embeddings):

        self.path_texts = path_texts # avg pool
        self.texts, self.splitted, self.max_seg_word = self.load_texts()
        self.mapping, self.vocab_id_dict, self.id_vocab_dict, self.vocab_emb_dict = self.load_embeddings(path_embeddings)
        self.size_emb = len(list(self.mapping["I"]))


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.texts[idx]
        splitted = self.splitted[idx]

        embeddings, id_embeddings = self.extract_embedding(splitted)

        sample = {"text": text, "embedding": embeddings, "id_embedding": id_embeddings}

        return sample

    def extract_embedding(self, splitted):

        embeddings = np.zeros((self.max_seg_word, self.size_emb), dtype = float)
        id_embeddings = np.zeros((self.max_seg_word, 1), dtype = float)

        index = 0

        for word in splitted:

            if len(word)>=1:

                word_upper = word.upper()

                if word_upper in self.vocab_emb_dict:

                    id_emb  = self.vocab_id_dict[word_upper]
                    id_embeddings[index,:] = id_emb

                    emb = list(self.vocab_emb_dict[word_upper])
                    embeddings[index,:] = emb

                index+=1

        return embeddings, id_embeddings


    def load_texts(self):

        with open(self.path_texts, "r", encoding='utf-8') as f:
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

        return  np.array(text), np.array(splitted), largest_split

    def load_embeddings(self,fname):

        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])

        dict_words = dict()

        vocab_id_dict = dict()

        vocab_emb_dict = dict()

        id_vocab_dict = dict()

        current_id = 0

        for k,v in data.items():

            embs = list(v)
            dict_words[k] = embs

            if k.isalpha():

                vocab_id_dict[k] = current_id
                vocab_emb_dict[k] = embs
                id_vocab_dict[current_id] = k

                current_id += 1

        return dict_words, vocab_id_dict, id_vocab_dict, vocab_emb_dict


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
