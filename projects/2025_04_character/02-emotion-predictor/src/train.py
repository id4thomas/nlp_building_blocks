import argparse
import json
import logging
import os
import sys
import traceback

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Utilities
from utils.data_utils import MessagesDataset
from utils.hf_argparser import HfArgumentParser

from peft import get_peft_model
from utils.peft_utils import get_lora_config, get_adalora_config, get_lora_save_param_dict

### Wandb logging
USE_WANDB = False 
if os.environ["WANDB_ENTITY"]:
	print("Initializing wandb with {}".format(os.environ['WANDB_ENTITY']))
	import wandb
	USE_WANDB = True

logger = logging.getLogger(__name__)

## SEED
import random
import numpy as np
def set_seed(seed=100):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	n_gpu = torch.cuda.device_count()
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)

def train(config: dict, save_trained: bool = True) -> None:
	#### Split Configs
	data_config = config["data"]
	train_config = config["training"]
	output_config = config["output"]
 
	run_name = train_config['run_name']

	#### Set Seed
	set_seed(train_config["seed"])

	#### Load Tokenizer
	### Load Tokenizer
	tokenizer = AutoTokenizer.from_pretrained(train_config["pretrained_model"])
	tokenizer.padding_side = "left"
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

	#### Load Data
	train_ds = MessagesDataset(
		data_dir = data_config["file"]["train"],
		tokenizer = tokenizer,
		tokenization_config = data_config['tokenization']
	)
	val_ds = MessagesDataset(
		data_dir = data_config["file"]["dev"],
		tokenizer = tokenizer,
		tokenization_config = data_config['tokenization']
	)
	print("Loaded Data Train {} Val {}".format(len(train_ds), len(val_ds)))

	#### Load Model
	model = AutoModelForCausalLM.from_pretrained(
		train_config["pretrained_model"],
		attn_implementation="flash_attention_2",
		torch_dtype = torch.bfloat16
		# attention_implemenation = "eager"
	)
	## resize
	train_peft_config = train_config.get('peft', None)
	if train_peft_config:
		if train_peft_config['method']=="lora":
			peft_config = get_lora_config(train_peft_config)
			model = get_peft_model(model, peft_config)
		elif train_peft_config['method']=="adalora":
			peft_config = get_adalora_config(train_peft_config)
			model = get_peft_model(model, peft_config)
		else:
			raise ValueError("Peft method {} not supported".format(train_peft_config['method']))

	#### Prepare Directories
	out_dir = os.path.join(output_config["weight_dir"], run_name)
	train_config["output_dir"] = out_dir
	print("out_dir : ", out_dir)

	effective_batch_size = train_config["per_device_train_batch_size"]*train_config["gradient_accumulation_steps"]*torch.cuda.device_count()
	print("Effective Batch Size:",effective_batch_size)

	# Trainer Based Training
	training_args = HfArgumentParser(TrainingArguments).parse_dict(train_config, allow_extra_keys = True)[0]

	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer, mlm=False, return_tensors="pt"
	)
 
	# Init wandb
	run = wandb.init(
		project=os.environ["WANDB_PROJECT"],
		entity=os.environ["WANDB_ENTITY"],
		config=config,
		name=run_name
	)

			
	trainer = Trainer(
		model = model,
		args = training_args,
		data_collator = data_collator,
		train_dataset = train_ds,
		eval_dataset = val_ds,
	)

	trainer.train()

	# Final Eval with Best Model
	trainer.evaluate(val_ds, metric_key_prefix = "final")

	if save_trained:
		#### Save Best Model
		best_dir = os.path.join(out_dir,"best")
		trainer.save_model(str(best_dir))
		tokenizer.save_pretrained(str(best_dir))
		with open(os.path.join(best_dir, "train_configs.json"), 'w') as f:
			json.dump(config, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_dir')
	args = parser.parse_args()
 
	with open(args.config_dir, "r") as f:
		config = json.load(f)

	try:
		train(config, save_trained = True)
	except Exception as e:
		traceback.print_exc()
		sys.exit(-1)
