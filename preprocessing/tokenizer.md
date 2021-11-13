# Tokenizer

* Tokenizing (splitting strings in sub-word token strings)
* encoding/decoding (i.e., tokenizing and converting to integers)

## Tokenization Process (Huggingface)
https://github.com/huggingface/transformers/blob/21546e59a6cd4a5c306a050109e0cea9bfdb3fb7/src/transformers/tokenization_utils_base.py#L1434

### GPT2 Tokenizer Example
https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/tokenization_gpt2.py
PreTrainedTokenizerBase -> PretrainedTokenizer -> GPT2Tokenizer

Tokenization Function Calls (Not Batched)
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
                * <b>build_inputs_with_special_tokens</b> (PreTrainedTokenizerBase)
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
| XLNet | \<s> | \</s> | \<sep> | \<cls> | \<unk> | \<pad> | \<mask> | [\<eop>,\<eod>] end of paragraph,document |

Additional Notes
* BERT: [CLS] token: weighted <b>average</b> of the words such that the representation <b>of the whole sequence</b> is captured. [2] Used for downstream tasks.
* GPT: needs ftfy,spacy package to use og paper tokenizer 
* RoBERTa: Use CLS,SEP as beginning & end of sequence
* T5: Use SEP as end of sequence
* XLNet: Use CLS,SEP as beginning & end of sequence

### Tokenized Format

build_inputs_with_special_tokens function of PreTrainedTokenizerBase performs no modifications<br>
(This implementation does not add special tokens and this method should be overridden in a subclass.)<br>
In case of RoBERTa, build_inputs_with_special_tokens is overridden<br>
(PreTrainedTokenizerBase -> PretrainedTokenizer -> GPT2Tokenizer -> Robertatokenzier)

if add_special_tokens=True (default:True) build_inputs_with_special_tokens is called<br>
(add_special_tokens: Whether or not to encode the sequences with the special tokens relative to their model.)

<!-- | Models   | Single Sequnce | Pair of Sequences| Position Embeddings |
| -------- | ------ | ------ | ------ |
| BART | \<s> X \</s> | \<s> A \</s>\</s> B \</s> | |
| BERT | [CLS] X [SEP] | [CLS] A [SEP] B [SEP] | |
| GPT | | | |
| GPT-2 | X | A B (token_ids_0 + token_ids_1) | |
| RoBERTa | \<s> X \</s> | \<s> A \</s>\</s> B \</s> | |
| T5 | X \</s> | | |
| XLNet | X \<sep>\<cls> | | | -->


Tokenizer Output Examples

Tokenizer \__call__ returns dictionary with keys [input_ids,attention_mask] and token_type_ids if applicable (Bert,XLNet)

| Models   | Tokenized | Notes |
| -------- | ------ | ------ |
| BART | \<s> X \</s> | |
| BERT | [CLS] X [SEP] | |
| GPT-2 | X | Padding token needs to be initialized |
| RoBERTa | \<s> X \</s> | |
| T5 | X \</s> | |
| XLNet | X \<sep>\<cls> | Padding is added to the left |

## References
[1] https://huggingface.co/transformers/ <br>
[2] https://discuss.huggingface.co/t/significance-of-the-cls-token/3180/7