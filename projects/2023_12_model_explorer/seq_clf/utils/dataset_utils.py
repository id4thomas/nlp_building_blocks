import torch
from torch.utils.data import Dataset

class PLMDataset(Dataset):
	def __init__(self, tokenizer, sources, seq_max_length = 512):
		self.tokenizer = tokenizer
		self.sources = sources
		self.encoded = tokenizer(
			sources, return_tensors = "pt", max_length = seq_max_length, truncation = True, padding = "max_length"
		)

	def __getitem__(self, idx):
		input_ids = self.encoded["input_ids"][idx]		
		attention_mask = self.encoded["attention_mask"][idx]		
		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask
		}
		
	def __len__(self):
		return len(self.sources)