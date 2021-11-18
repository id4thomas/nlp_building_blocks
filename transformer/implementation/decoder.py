import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *

class DecoderBlock(nn.Module):
    def __init__(self,config):
        super(DecoderBlock,self).__init__()
        
        self.d_model=config["d_model"]

        #Multi Head Attention
        self.self_mha=MultiHeadAttention(config)
        self.encdec_mha=MultiHeadAttention(config)

        #FFNN
        self.ff=PositionWiseFeedForward(config)

        #Add and Normalize
        self.addandnorm=AddAndNorm(config)

    def forward(self,x,memory):
        #1- Multi-Head Self Attention
        self_att=self.self_mha(x,x,x)
        
        #2- Add & Norm
        self_att=self.addandnorm(x,self_att)

        #3- Multi-head Encoder-Decoder Attention
        #Key, Value: Encoder Output
        enc_dec_att=self.encdec_mha(self_att,memory,memory)

        #4- Add & Norm
        enc_dec_att=self.addandnorm(self_att,enc_dec_att)

        #5- FFNN
        ff_out=self.ff(enc_dec_att)

        #6- Add & Norm
        ff_out=self.addandnorm(enc_dec_att,ff_out)

        return ff_out