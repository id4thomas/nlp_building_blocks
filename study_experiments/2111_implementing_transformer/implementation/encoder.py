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

        #FFNN
        self.ff=PositionWiseFeedForward(config)

        #Add and Normalize
        self.addandnorm=AddAndNorm(config)

    def forward(self,x):
        pass
        #1- Multi-Head Attention
        qkv=self.mha(x,x,x)

        #2- Add & Norm
        #Dropout
        att=self.addandnorm(x,qkv)

        #3- FFNN
        ff_out=self.ff(att)

        #4- Add & Norm
        ff_out=self.addandnorm(att,ff_out)

        return ff_out