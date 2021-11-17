import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *

class EncoderBlock(nn.Module):
    def __init__(self,config):
        super(EncoderBlock,self).__init__()
        
        self.d_model=config["d_model"]

        #Multi Head Attention
        self.mha=MultiHeadAttention(config)

        #Add&Norm
        #Dropout & LayerNorm
        self.att_dropout=nn.Dropout(p=config["dropout_prob"])
        self.att_layernorm=nn.LayerNorm(self.d_model)

        #FFNN
        self.ff_w1=nn.Linear(self.d_model,config["d_ff"])
        self.ff_w2=nn.Linear(config["d_ff"],self.d_model)

        #Dropout & LayerNorm
        self.ff_dropout=nn.Dropout(p=config["dropout_prob"])
        self.ff_layernorm=nn.LayerNorm(self.d_model)

    def forward(self,x):
        pass
        #1- Multi-Head Attention
        qkv=self.mha(x)

        #2- Add & Norm
        #Dropout
        qkv=self.att_dropout(qkv)

        #Residual Add
        att=qkv+x

        #LayerNorm
        att=self.att_layernorm(att)

        #3- FFNN
        ff_out=self.ff_w2(F.relu(self.ff_w1(att)))

        #4- Add & Norm
        ff_out=self.ff_dropout(ff_out)

        ff_out=ff_out+att

        ff_out=self.ff_layernorm(ff_out)

        return ff_out