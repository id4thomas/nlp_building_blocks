import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):

    def __init__(self,d_model,d_k):
        super(ScaledDotProductAttention,self).__init__()

        self.d_k=d_k
        self.d_k_sqrt=math.sqrt(self.d_k)
        
        #3 Weights
        self.w_q=nn.Linear(d_model,self.d_k,bias=False)
        self.w_k=nn.Linear(d_model,self.d_k,bias=False)
        self.w_v=nn.Linear(d_model,self.d_k,bias=False)

        self.softmax=nn.Softmax(dim=-1)

    def forward(self,query,key,value):
        q=self.w_q(query)
        k=self.w_k(key)
        v=self.w_v(value)

        #1- QK^T
        qk=torch.bmm(q,torch.transpose(k,1,2))  #Batch Matrix Multiplication (b,m,*)*(b,*,n)-> (b,m,n)

        #2 - Scale: div(sqrt(d_key)) -> Softmax
        att=self.softmax(torch.div(qk,self.d_k_sqrt))
        
        #3 *V
        qkv=torch.bmm(att,v)

        return qkv


class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super(MultiHeadAttention,self).__init__()
        
        self.num_heads=config["num_heads"]
        self.d_model=config["d_model"]
        self.d_k=int(self.d_model/self.num_heads)

        self.heads={}
        for i in range(self.num_heads):
            self.heads[f"head_{i}"]=ScaledDotProductAttention(self.d_model,self.d_k)

        #Projection Layer
        self.att_proj=nn.Linear(self.d_model,self.d_model)
        

    def forward(self,query,key,value):
        # Run through all heads
        qkv_res=[]

        for i in range(self.num_heads):
            qkv_res.append(self.heads[f"head_{i}"](query,key,value))
        
        #Concat & Project
        qkv=self.att_proj(torch.cat(qkv_res,dim=-1))    

        return qkv

class PositionWiseFeedForward(nn.Module):
    def __init__(self,config):
        super(PositionWiseFeedForward,self).__init__()

        self.w1=nn.Linear(config["d_model"],config["d_ff"])
        self.w2=nn.Linear(config["d_ff"],config["d_model"])

        if config["actv"]=="relu":
            self.actv=nn.ReLU()
        elif config["actv"]=="gelu":
            self.actv=nn.GeLU()
        else:
            self.actv=nn.ReLU()

        #Dropout
        self.dropout=nn.Dropout(p=config["dropout_prob"])

    def forward(self,x):
        #X -> W1 -> Actvation -> Dropout -> W2
        return self.w2(self.dropout(self.actv(self.w1(x))))
        
        
class AddAndNorm(nn.Module):
    def __init__(self,config):
        super(AddAndNorm,self).__init__()

        self.dropout=nn.Dropout(p=config["dropout_prob"])
        self.layernorm=nn.LayerNorm(config["d_model"])

    def forward(self,x, layer_out):
        return self.layernorm(x+self.dropout(layer_out))