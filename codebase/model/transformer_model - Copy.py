# Image Model class

import argparse
from typing import Any, Dict
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

D_MODEL = 512
HEADS = 8
NUM_LAYERS = 3



# new pos encoder
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    """Container module with an encoder, a  transformer module, and a decoder."""
    def __init__(
        self,
        data_config: Dict[str, Any] = None,
        args: argparse.Namespace = None,
    ) -> None:

        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        self.d_model = self.args.get("d_model", D_MODEL)
        self.num_layers = self.args.get("num_layers", NUM_LAYERS)
        self.heads = self.args.get("heads", HEADS)
        self.vocab_size = self.data_config['num_tokens']

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.heads, dim_feedforward=self.d_model, 
                                          num_encoder_layers=self.num_layers,num_decoder_layers=self.num_layers,
                                          batch_first = True)
        self.logit = nn.Linear(self.d_model, self.vocab_size)

        #self.src_mask = None
        
        #self.init_weights()


        #def init_weights(self):
        #  initrange = 0.1
        # nn.init.uniform_(self.embd.weight, -initrange, initrange)
        #  nn.init.zeros_(self.decoder.bias)
        #  nn.init.uniform_(self.decoder.weight, -initrange, initrange)



    def forward(self, src,tgt,tgt_mask):
        
        src = self.embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src=src,tgt=tgt,tgt_mask=tgt_mask)
        output = self.logit(output)
        #return F.log_softmax(output, dim=-1)
        return output
    
    @staticmethod
    def add_to_argparse(parser):
       parser.add_argument("--d_model", type=int, default=D_MODEL, help="embedding dimension")
       parser.add_argument("--heads", type=int, default=HEADS, help="number of transformer head")
       parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help="number of transformer layer")
       return parser

"""
class resnet18(nn.Module):
    

    def __init__(
        self,
        data_config: Dict[str, Any] = None,
        args: argparse.Namespace = None,
    ) -> None:

        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        

    def forward(self, x):
        x = self.model(x)
        return x
"""
    