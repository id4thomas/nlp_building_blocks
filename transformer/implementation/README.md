# Transformer Implementation

```
└── attention.py
    └── MultiHeadAttention
    └── ScaledDotProductAttention
    └── PositionWiseFeedForward
    └── AddAndNorm
└── encoder.py
    └── EncoderBlock
└── decoder.py
    └── DecoderBlock
└── my_transformer.py
    └── Transformer
```

Configuration Dictionary for the transformer model is as following.
```
config={
    'd_model':512,
    'd_ff':2048,
    'num_heads':8,
    'num_encoder_blocks':2,
    'num_decoder_blocks':2,
    'dropout_prob':0.1,
    'debug':True,
    'actv':"relu"
}
```
## attention.py
### MultiHeadAttention
Initializes <i>num_heads</i> number of ScaledDotProduct Attention instances<br>
Concatenates attention results and projects with <i>att_proj</i> layer.<br>
<i>num_heads</i>*<i>dim_k</i>=><i>dim_model</i>

### ScaledDotProductAttention
1. Project with <i>w_q</i>,<i>w_k</i>,<i>w_V</i> to Q, K, V
2. QK<sup>T</sup> by torch.bmm (batch matrix multiplication)
3. Scale by sqrt(d_key) and apply softmax
4. Att*V by torch.bmm


### PositionWiseFeedForward
1. Applies W1 projection (into larger dimension than <i>d_model</i>)
2. Apply activation function (ReLU for transformer, GeLU for BERT,..)
3. Apply dropout to activated output
4. Apply W2 projection

### AddAndNorm
1. Applies dropout to sublayer output
2. Residual add sublayer input & output
3. Apply Layer Normalization

## EncoderBlock - encoder.py
1. Apply Multi head self-attention
2. Add & Normalize
3. PositionWise Feed Forward
4. Add & Normalize

## DecoderBlock - decoder.py
1. Apply Multi head self-attention
2. Add & Normalize
3. Apply Multi head encoder-decoder-attention
    * Use encoder output as Key & Value (Query is self-attention output)
4. Add & Normalize
3. PositionWise Feed Forward
4. Add & Normalize


