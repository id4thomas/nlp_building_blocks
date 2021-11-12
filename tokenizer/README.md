# Tokenizer

* Tokenizing (splitting strings in sub-word token strings)
* encoding/decoding (i.e., tokenizing and converting to integers)

## Tokenization Process (Huggingface)
https://github.com/huggingface/transformers/blob/21546e59a6cd4a5c306a050109e0cea9bfdb3fb7/src/transformers/tokenization_utils_base.py#L1434

### GPT2 Tokenizer Example
PreTrainedTokenizerBase -> PretrainedTokenizer -> GPT2Tokenizer

Tokenization Flow (Not Batched)
* \__call__ (PreTrainedTokenizerBase) 
    * encode_plus (PreTrainedTokenizerBase)
        * _get_padding_truncation_strategies (PretrainedTokenizer)
        * \_encode_plus (PretrainedTokenizer)
            * get_input_ids (PretrainedTokenizer)
                * tokenize (PretrainedTokenizer)
                    * _tokenize (GPT2Tokenizer)
                        * bpe (GPT2Tokenizer)
                * convert_tokens_to_ids (PretrainedTokenizer)
                    * _convert_token_to_id_with_added_voc (PretrainedTokenizer)
                        * _convert_token_to_id (GPT2Tokenizer)
                            * self.encoder.get(token) (GPT2Tokenizer)
            * prepare_for_model (PreTrainedTokenizerBase)
                * truncate_sequences (PreTrainedTokenizerBase)
                * pad (PreTrainedTokenizerBase)
                    * _pad (PreTrainedTokenizerBase)

    

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

Input Format
| Models   | Single Sequnce | Pair of Sequences|
| -------- | ------ | ------ |
| BART | \<s> X \</s> | \<s> A \</s>\</s> B \</s> |
| BERT | [CLS] X [SEP] | [CLS] A [SEP] B [SEP] |
| GPT | | |
| GPT-2 | | |
| RoBERTa | \<s> X \</s> | \<s> A \</s>\</s> B \</s> |
| T5 | | |
| XLNet | | |

## References
https://huggingface.co/transformers/