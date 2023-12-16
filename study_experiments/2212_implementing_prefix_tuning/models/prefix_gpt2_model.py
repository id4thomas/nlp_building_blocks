import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# 정상적 generate 위해 prepare_inputs_for_generation 부분을 조정해서 정상적으로
class myGPT2LMHeadModel(GPT2LMHeadModel):
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs

        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        ##############################
        if past is None:
            # print('only at the beginnning. ')
            if 'past_key_values' in kwargs:
                past = kwargs['past_key_values']
            else:
                past = None

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None

        # PREFIX 길이 만큼 attention_mask를 조정해준다.
        # ex. prefix 길이 8, input_ids 길이 26이라면
        # attention_mask는 34, position_ids는 26이 되어야함.
        # attention_mask는 query*key^T 곱 이후에 반영되기 때문에 prefix 길이 필요
        if past is not None and attention_mask is not None:
            prefix_len = past[0].shape[-2]
            bsz = input_ids.shape[0]
            device = attention_mask.device
            prefix_attention_mask = torch.ones(bsz, prefix_len)
            attention_mask = attention_mask.detach().cpu()
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim = -1).to(device)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

# class PrefixGPT2LMHeadModel(PushToHubFriendlyModel):
class PrefixGPT2LMHeadModel(nn.Module):
    main_input_name = "input_ids"
    def __init__(self, args):
        super().__init__()
        self.args = args
        """The prefix-tuning code"""

        self.preseqlen = args.prefix_sequence_length
        self.mid_dim = args.mid_dim

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        # self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_location, use_fast=False)
        # KoGPT2 special_tokens 반영해서 로드
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_location,
                    bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                    pad_token='<pad>', mask_token='<mask>')

        self.pretrain_model = myGPT2LMHeadModel.from_pretrained(
            args.pretrained_location
        )
        self.config = self.pretrain_model.config

        # Parameter follows skt/kogpt2-base-v2
        self.match_n_layer = self.config.n_layer
        self.match_n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head # huggingface BART's dim of kv need to be calculated

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        self.dropout = nn.Dropout(args.prefix_dropout)
        
        #### PARAMETER FREEZE
        if self.args.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if self.args.freeze_prefix:
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.control_trans.parameters():
                param.requires_grad = False

    def get_prompt(self, bsz = None, sample_size=1):
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz
        )

        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_prompt,
            **kwargs,
        ).loss
        return {'loss': loss, "logits": outputs.logits}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams']
        )
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_prompt,
            use_cache=False, # Important! use_cache needs to be false, we provide our custom past
            **kwargs,
        )

        return generated_ids