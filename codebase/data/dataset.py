# dataset class 
from collections import Counter
import json
import torch
from torch.utils.data import Dataset
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)


from base_data_module import BaseDataModule, load_and_print_info

# Directory for downloading and storing data
DATA_DIRNAME = BaseDataModule.data_dirname() 

MAX_LEN = 25  # max length of the sequence
MIN_WORD_FREQ = 5 # minimum nr of times a word to be appeared for token consideration



class datamodule(BaseDataModule):
    """Image DataModule."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self.max_len = MAX_LEN + 2
        self.num_tokens = 0
        self.mapping = {}
        self._prepare_data()

    def _remove_punc(self,string):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        no_punct = ""
        for char in string:
            if char not in punctuations:
                no_punct = no_punct + char  # space is also a character
        return no_punct.lower()

    
    def _encode_question(self,words, word_map):
        enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (MAX_LEN - len(words))
        return enc_c
    
    def _encode_reply(self,words, word_map):
        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + \
        [word_map['<end>']] + [word_map['<pad>']] * (MAX_LEN - len(words))
        return enc_c

    
    def _prepare_data(self, *args, **kwargs) -> None:
        """Prepare data"""

        DATA_DIRNAME/'hymenoptera_data'/'train'
        corpus_movie_conv = DATA_DIRNAME/'movie_conversations/movie_conversations.txt'
        corpus_movie_lines = DATA_DIRNAME/'movie_lines/movie_lines.txt'

        with open(corpus_movie_conv, 'r') as c:
            conv = c.readlines()

        with open(corpus_movie_lines, 'r',encoding='iso-8859-1') as l:
            lines = l.readlines()

        lines_dic = {}
        for line in lines:
            objects = line.split(" +++$+++ ")
            lines_dic[objects[0]] = objects[-1]


        pairs = []
        for con in conv:
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):
                qa_pairs = []
                
                if i==len(ids)-1:
                    break
                
                first = self._remove_punc(lines_dic[ids[i]].strip())      
                second = self._remove_punc(lines_dic[ids[i+1]].strip())
                qa_pairs.append(first.split()[:MAX_LEN])
                qa_pairs.append(second.split()[:MAX_LEN])
                pairs.append(qa_pairs)



        word_freq = Counter()
        for pair in pairs:
            word_freq.update(pair[0])
            word_freq.update(pair[1])

        words = [w for w in word_freq.keys() if word_freq[w] > MIN_WORD_FREQ]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

        self.mapping = word_map
        self.num_tokens = len(self.mapping)

        #with open('WORDMAP_corpus.json', 'w') as j:
        #   json.dump(word_map, j)

        pairs_encoded = []
        for pair in pairs:
            qus = self._encode_question(pair[0], word_map)
            ans = self._encode_reply(pair[1], word_map)
            pairs_encoded.append([qus, ans])
        
        with open(DATA_DIRNAME/'pairs_encoded.json', 'w') as p:
            json.dump(pairs_encoded, p)
    


    def setup(self, stage=None) -> None:
   
        data = json.load(open(DATA_DIRNAME/'pairs_encoded.json'))

        train, test = train_test_split(data, test_size=0.2,random_state=42)
        val, test = train_test_split(test, test_size=0.5,random_state=42)

        self.data_train:torch.utils.data.Dataset = TextDataset(data = train)
        self.data_val:torch.utils.data.Dataset = TextDataset(data = val)
        self.data_test:torch.utils.data.Dataset = TextDataset(data = test)
    
    
    # change according to dataset properties 
    def __repr__(self):
        basic = f"Text Dataset\nNum words: {self.num_tokens}\nMax Length: {self.max_len}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype,)}\n"
            f"Batch y stats: {(y.shape, y.dtype,)}\n"
        )
        return basic + data
    


class TextDataset(Dataset):

    def __init__(self,data):

        self.pairs = data
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
            
        return question, reply

    def __len__(self):
        return self.dataset_size

if __name__ == "__main__":
    load_and_print_info(datamodule)








