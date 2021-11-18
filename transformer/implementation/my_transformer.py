import torch
import torch.nn as nn

from encoder import *
from decoder import *

class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer,self).__init__()

        #Encoder
        self.num_encoder_blocks=config["num_encoder_blocks"]
        self.encoder_blocks={}
        for i in range(self.num_encoder_blocks):
            self.encoder_blocks[f"enc_{i}"]=EncoderBlock(config)

        #Decoder
        self.num_decoder_blocks=config["num_decoder_blocks"]
        self.decoder_blocks={}
        for i in range(self.num_decoder_blocks):
            self.decoder_blocks[f"dec_{i}"]=DecoderBlock(config)

    
    def forward(self,x):
        #Encoder
        for i in range(self.num_encoder_blocks):
            x=self.encoder_blocks[f"enc_{i}"](x)

        memory=x

        #Decoder
        for i in range(self.num_decoder_blocks):
            x=self.decoder_blocks[f"dec_{i}"](x,memory)
        
        return x


class TransformerLM(nn.Module):
    def __init__(self,config):
        super(TransformerLM,self).__init__()
    
    def forward(self,x):
        pass
