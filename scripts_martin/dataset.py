import os
import re
import io
import string
import glob
import linecache
import itertools
import torch
import numpy as np
import kaldiio
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """Loads a how2 audio-video-text dataset
        :param HOW2_PATH: base directory of how2-dataset
        :param subfolder: dataset split (train, test, val)
    """
    def __init__(self, HOW2_PATH, subfolder):
        if subfolder not in ['train', 'val', 'test']:
            raise ValueError(f'subfolder must be either train, test or val ({subfolder} given)')

        # Set relative paths for 300h
        self.subfolder = ('dev5' if subfolder == 'test' else subfolder)
        self.PATH_BASE = HOW2_PATH + f'how2-300h-v1/data/{self.subfolder}/'
        self.PATH_RELEASE = HOW2_PATH + 'how2-release/'

        ## Feature lookup
        self.PATH_FEATURES_A = HOW2_PATH + 'fbank_pitch_181506/'
        self.PATH_FEATURES_V = HOW2_PATH + f'resnext101-action-avgpool-300h/'
        self.PATH_FEATURES_T = self.PATH_RELEASE + 'word_embedding/cmu_partition.train.vec'

        # Video segment lookups
        print("Loading video segments...  ", end="", flush=True)
        self.segments = self.load_segments()
        print('OK')

        print("Building segment/video reverse search...  ", end="", flush=True)
        self.seg2vid, self.vid2segs, self.vid2textvid = self.load_reversesearch()
        print('OK')

        print("Building embeddings lookup...  ", end="", flush=True)
        self.textemb_lines, self.cmvnpaths = self.load_emb_lines()
        print('OK')

        # Video features lookup
        print("Loading video embeddings memmap...  ", end="", flush=True)
        self.video_features = np.load(self.PATH_FEATURES_V + f'{self.subfolder}.npy', mmap_mode='r')
        print('OK') 

    def __len__(self):
        return len(self.segments)

    def load_segments(self):
        """Loads list of video segments in current dataset split 
            :returns: list of segments in current split
        """
        segments = []
        with open(self.PATH_BASE + 'segments', 'r', encoding='utf-8') as f:
            # line format: segment_id video_id start_time end_time
            for line in f:
                segments.append(re.findall(r'\S+', line)[0])
        return segments
    
    def load_reversesearch(self):
        """Builds searching dictionnaries for fast indexing
            :returns: seg2vid mapping segment id XXXXXXXXXXX_Y to its video id XXXXXXXXXXX
            :returns: vid2segs mapping video id XXXXXXXXXXX to list of its segment ids [XXXXXXXXXXX_Y]
            :returns: vid2textid mapping video id XXXXXXXXXXX to vidYYYY
        """
        # segment to video mapping
        # example --8pSDeC-fg_0 --> --8pSDeC-fg
        print("seg2vid...  ", end="", flush=True)
        seg2vid = dict()
        with open(self.PATH_BASE + 'utt2spk', 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 1:
                    continue
                seg, vid = re.findall(r'\S+', line)
                seg2vid[seg] = vid
        # seg2vid OK

        # list of segments per video
        # example --8pSDeC-fg --> [--8pSDeC-fg_0 --8pSDeC-fg_1 ...]
        print("vid2segs...  ", end="", flush=True)
        vid2segs = dict()
        with open(self.PATH_BASE + 'spk2utt', 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 1:
                    continue
                segs = re.findall(r'\S+', line)
                vid = segs.pop(0)
                seg2vid[vid] = segs
        # vid2segs OK

        # video ids for text embeddings
        # example --8pSDeC-fg --> vid00002
        print("vid2textid...  ", end="", flush=True)
        vid2textid = dict()
        with open(self.PATH_RELEASE + f'info/video_ids/{self.subfolder}_video_ids.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 1:
                    continue
                vid, textvid = re.findall(r'\S+', line)
                vid2textid[vid] = textvid
        # vid2textid OK

        return seg2vid, vid2segs, vid2textid
    
    def seg2textid(self, seg):
        """Gets video id (vidXXXXX-YYYY format) from segment id (XXXX_Y format). Useful because text embeddings are mapped with video id 
            :param seg: segment id in (XXXX_Y format)
            :returns: segment id in (vidXXXXX-YYYY format)
        """
        vid = self.seg2vid[seg]
        segid = int(seg.split('_')[-1]) + 1
        new_vid = self.vid2textvid.get(vid, 'vidXXXXX')

        return new_vid + '-' + '{:04d}'.format(segid)

    # not all vid00002-0001 have a vector ==> map to default 

    def load_emb_lines(self):
        """Reads the text embedding file and maps keys to line numbers to avoid charging whole embeddings in memory
            :returns: dictionary mapping key: line_number in embedding file
            :returns: dictionary mapping key: line_number in embedding file
        """
        # Maps line of text embeddings file to avoid charging all in memory
        print("text embeddings...  ", end="", flush=True)
        textlines = dict()
        with open(self.PATH_FEATURES_T, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                # keep only line id: word of videoid and save line number to not to load whole vec in memory
                lineid = [item.group() for item in itertools.islice(re.finditer(r'\S+', line), 1)][0]
                textlines[lineid] = i + 1
        

        print("audio embeddings...  ", end="", flush=True)
        cmvnpaths = dict()
        with open(self.PATH_BASE + 'cmvn.scp', 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 1:
                        continue
                vid, arkpath = re.findall(r'\S+', line)
                cmvnpaths[vid] = arkpath
        return textlines, cmvnpaths
    
    def get_text_embedding(self, key, zeros=True):
        """Gets the embedding vector associated with a video id
            :param key: video id (vidXXXXX-YYYY format) or word to recover the embedding of
            :param zeros: whether to return an empty vector or the embedding of stop mark </s>
            :returns: numpy array of length 100 containing the embedding
        """
        line_id = self.textemb_lines.get(key, 0)

        if line_id == 0:
            if zeros:
                return np.zeros(100)
            else:
                # remap to default word
                key = '</s>'
                line_id = self.textemb_lines.get(key)

        # get line corresponding to vector
        vec_line = linecache.getline(self.PATH_FEATURES_T, line_id)
        # drop key to have 100-dim vector
        vec = vec_line.replace(key, '')
        # return vector
        return np.fromstring(vec, dtype=np.float64, sep=' ')
    
    def get_text_string(self, idx, lang='en'):
        """Gets the text associated with a video segment
            :param segid: segment id (XXXX_Y format)
            :param lang: language of text (en or pt)
            :returns: string of text in segment
        """
        if lang not in ['en', 'pt']:
            raise ValueError(f'language must be "en" or "pt" ({lang} given)')
        
        # Raw texts file
        fname = self.PATH_BASE + f'text.{lang}'
        # Access by line number
        return linecache.getline(fname, idx+1).strip()
    
    def tokenize(self, text):
        """Upper-case and remove punctuation from text
            :param text: string to tokenize
            :returns: a list of tokens
        """
        punctuation = '!()-[]{};:"\,<>./?@#$%^&*_~'
        return text.upper().translate(str.maketrans('', '', punctuation)).split()

    def get_audio(self, idx, cmvn=True):
        """Returns kaldiio-matrix of a segment
            :param idx:
            :param cmvn: whether to return or not the Cepstral mean variance normalization (CMVN) of the segment's audio
            :returns: tensor containing audio features (if cmvn then shape [2,43])
        """
        # get segment_id
        segment = self.segments[idx]
        # if CMVN read path directly
        if cmvn:
            video = self.seg2vid[segment]
            ark_path = self.cmvnpaths[video]
        # else read segment_id line from scp
        else:
            scp_file = self.PATH_BASE + 'feats.scp'
            ark_path = linecache.getline(scp_file, idx+1).strip().split()[1]
        # adapt path to dataset
        ark_path = ark_path.replace('ARK_PATH/', self.PATH_FEATURES_A)
        # load matrix
        matrix = kaldiio.load_mat(ark_path)
        # Get CMVN transform by dropping last col (src: https://pykaldi.github.io/_modules/kaldi/transform/cmvn.html)    
        if cmvn:
            matrix = matrix[:,:-1]
        return torch.from_numpy(matrix)
    
    def get_video(self, idx):
        """Returns pre-processed (resnext101-action) video embedding of a segment
            :param idx: index of segment
            :returns: video segment feature (size [2048])
        """
        # read from memory map
        feature_i = self.video_features[idx,:]
        # torch loads tensor in memory
        return torch.from_numpy(feature_i)

    def get_text(self, idx):
        """Returns text embedding of a segment
            :param idx: index of segment
            :returns: pre-trained text embedding (size [100])
        """

        segid = self.segments[idx]
        segid_old = self.seg2textid(segid)

        vec = self.get_text_embedding(segid_old, zeros=True)

        # Case when video doesn't have embedding: return mean of word embeddings in text
        if np.sum(vec) < 1e-9:
            # Recover raw text
            seg_text = self.get_text_string(idx, lang='en')
            # Tokenize
            tokens = self.tokenize(seg_text)
            # Keep track of valid embedded words
            valid_tokens = 0
            for tok in tokens:
                tok_vec = self.get_text_embedding(tok, zeros=True)
                vec += tok_vec
                if np.sum(tok_vec) < 1e-9:
                    valid_tokens += 1
            # Normalize
            if valid_tokens != 0:
                vec /= 1.0*valid_tokens
            # If no word has embedding, return embedding of </s>
            else:
                vec = self.get_text_embedding(segid_old, zeros=False)
        
        return torch.from_numpy(vec)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Recover segment id
        segment = self.segments[idx]
        # Load features
        audio = self.get_audio(idx, cmvn=True)
        video = self.get_video(idx)
        text = self.get_text(idx)
        # make sample
        sample = {"id": segment, "audio": audio, "video": video, "text": text}

        return sample


if  __name__ == "__main__":
    dataset = MultimodalDataset('../data/how2-dataset/', 'train')

    print(dataset.PATH_BASE)