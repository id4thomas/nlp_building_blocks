from peft import LoraConfig, AdaLoraConfig
from peft import get_peft_model, get_peft_model_state_dict, TaskType

def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)

### Peft Configs
def get_lora_config(config):
	return LoraConfig(
		task_type=TaskType.CAUSAL_LM, 
		inference_mode=False, 
		r = config["lora_r"], 
		lora_alpha = config["lora_alpha"], 
		lora_dropout = config["lora_dropout"],
		bias="none",
		target_modules= config['target_modules']
	)

def get_adalora_config(config):
	return AdaLoraConfig(
		task_type=TaskType.CAUSAL_LM, 
		inference_mode=False, 
		init_r = config["init_r"], 
		target_rank = config["target_rank"], 
		lora_alpha = config["lora_alpha"], 
		lora_dropout = config["lora_dropout"],
		bias="none",
		target_modules= config['target_modules']
	)

### Peft Configs
def get_lora_save_param_dict(model, save_embedding = False):
	state_dict = model.state_dict()
	params_to_save = get_peft_model_state_dict(model, state_dict=state_dict)
	
	if save_embedding:
		layer_keys = list(state_dict.keys())
		# embed_keys = list(filter(lambda x: "embed_in" in x, layer_keys))
		embed_keys = list(filter(lambda x: ("wte" in x) or ("wpe" in x), layer_keys))
		for k in embed_keys:
			params_to_save[k] = state_dict[k]
			
	return params_to_save

def get_embedding_dict(model):
	state_dict = model.state_dict()
	layer_keys = list(state_dict.keys())
	params_to_save = {}
	embed_keys = list(filter(lambda x: ("wte" in x) or ("wpe" in x), layer_keys))
	for k in embed_keys:
		params_to_save[k] = state_dict[k]
	return params_to_save