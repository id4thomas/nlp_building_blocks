# Tokenizer

## Special Tokens
Organize the special tokens for each LM Models. (Based on huggingface transformers documentation)

| Models   | BOS    | EOS | SEP | CLS | UNK | PAD | MASK | Additional | Notes |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----- | ------------- |
| BART | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | | |
| BERT | - | - | [SEP] | [CLS] | [UNK] | [PAD] | [MASK] | | |
| ELECTRA | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | | Identical to BERT Tokenizer |
| GPT | - | - | - | - | \<unk> | - | - | | needs ftfy,spacy package to use og paper tokenizer |
| GPT-2 | \<\|endoftext\|> | \<\|endoftext\|> | - | - | \<\|endoftext\|> | - | - | | |
| RoBERTa | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | | Use CLS,SEP as beginning & end of sequence |
| T5 | - | \</s> | - | - | \<unk> | \<pad> | - | \<extra_id_{}> | Use SEP as end of sequence |
| XLNet | \<s> | \</s> | \</s> | \<s> | \<unk> | \<pad> | \<mask> | [\<eop>,\<eod>] end of paragraph,document | Use CLS,SEP as beginning & end of sequence |

## References
https://huggingface.co/transformers/