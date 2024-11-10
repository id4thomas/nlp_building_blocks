import os
import json
import torch

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

try:
	from utils.dataset_utils import PLMDataset
except ModuleNotFoundError as e:
	from seq_clf.utils.dataset_utils import PLMDataset
except Exception as e:
	raise e

def load_model_and_tokenizer(model_dir, device = "cpu", model_type = "classifier"):
	if model_type=="classifier":
		model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	else:
		model = AutoModel.from_pretrained(model_dir)
	model.eval()
	model.to(device)
	print("LOADED MODEL")

	try:
		tokenizer = AutoTokenizer.from_pretrained(model_dir)
	except Exception as e:
		with open(os.path.join(model_dir, "config.json"), "r") as f:
			config = json.load(f)
		tokenizer = AutoTokenizer.from_pretrained(config["_name_or_path"])
	return model, tokenizer

## Extracting Hidden Latents
@torch.no_grad()
def extract_hid(model, tokenizer, sources, device = "cpu", batch_size = 32, seq_max_length = 512):
	dataset = PLMDataset(tokenizer, sources, seq_max_length = seq_max_length)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	all_output = []
	for batch in tqdm(dataloader):
		# Load on Device
		# detectgpt doesn't provide attention_mask
		output = model(
			batch["input_ids"].to(device),
			attention_mask = batch["attention_mask"].to(device)
		)

		hidden = output[0].detach().cpu()
		# print(hidden.shape)
		all_output.append(hidden)

	all_hidden = []
	for h in all_output:
		# Hidden dim of 1st token
		all_hidden.append(h[:, 0, :] )
	hidden = torch.cat(all_hidden, dim = 0)
	return hidden


## Predict
@torch.no_grad()
def inference(model, dataloader, device = "cpu"):
	logits = None
	for batch in tqdm(dataloader):
		# Load on Device
		# inputs = data.to(device)
		# Forward
		with torch.no_grad():
			outputs = model(
				batch["input_ids"].to(device),
				attention_mask = batch["attention_mask"].to(device)
			)
		# Append Logits
		if logits is not None:
			logits = torch.cat((logits, outputs.logits.cpu()))
		else:
			logits = outputs.logits.cpu()
	return torch.nn.functional.softmax(logits)

def predict(model, tokenizer, sources, device = "cpu", batch_size = 32, seq_max_length = 512):
	dataset = PLMDataset(tokenizer, sources, seq_max_length = seq_max_length)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	logits = inference(model, dataloader, device = device)

	results = []
	for _logits in logits:
		# Label
		label = _logits.argmax().item()
		# Probability (Needs Update Later..)
		prob = _logits.numpy()
		fake_prob = _logits[1].item()

		results.append({
			"label": label,
			"label_prob": prob,
			"fake_prob": fake_prob
		})

	return results