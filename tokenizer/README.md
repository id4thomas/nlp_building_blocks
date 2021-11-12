# Tokenizer

## Special Tokens
Organize the special tokens for each LM Models. (Based on huggingface transformers documentation)

| Models   | BOS    | EOS | SEP | CLS | UNK | PAD | MASK | Additional | 
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----- | 
| BART | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | |
| BERT | - | - | [SEP] | [CLS] | [UNK] | [PAD] | [MASK] | |
| ELECTRA | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | |
| GPT | - | - | - | - | \<unk> | - | - | |
| GPT-2 | \<\|endoftext\|> | \<\|endoftext\|> | - | - | \<\|endoftext\|> | - | - | |
| RoBERTa | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | 
| T5 | - | \</s> | - | - | \<unk> | \<pad> | - | \<extra_id_{}> | 
| XLNet | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | [\<eop>,\<eod>] end of paragraph,document |

Additional Notes
* GPT: needs ftfy,spacy package to use og paper tokenizer 
* RoBERTa: Use CLS,SEP as beginning & end of sequence
* T5: Use SEP as end of sequence
* XLNet: Use CLS,SEP as beginning & end of sequence

## References
https://huggingface.co/transformers/