import torch
import random


def preprocess_function(examples, tokenizer, max_length=256):
	dataset_transform = lambda xs: xs["text"]
	all_tokenized = [tokenizer.encode(t, return_tensors="pt") for t in dataset_transform(examples)]
	new_tokenized = [{"input_ids": t} for t in all_tokenized]
	for i, t in enumerate(new_tokenized):
		new_tokenized[i]["input_ids"] = truncate_or_pad(t['input_ids'], tokenizer.pad_token_id, max_length=max_length)
	new_input_ids = torch.cat([t["input_ids"] for t in new_tokenized], dim=0)
	new_attention_mask = (new_input_ids != tokenizer.pad_token_id).long()
	tokenized = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
	tokenized["labels"] = tokenized["input_ids"].clone()
	return tokenized

def truncate_or_pad(t, padding_idx=0, max_length=256):
	if t.shape[1] > max_length:
		start = random.randint(0, t.shape[1] - max_length)
		t = t[:, start:start + max_length]
	else:
		padding = torch.zeros(t.shape[0], max_length - t.shape[1], dtype=t.dtype, device=t.device)
		t = torch.cat([t, padding + padding_idx], dim=1)
	return t
