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


def _format_csqa_prompt(question, choices):
	# Matches the reference impl format (eval_helpers.py from ezelikman/quiet-star).
	choice_lines = "\n".join(
		f"({label}) {text}" for label, text in zip(choices["label"], choices["text"]))
	return f"Q: {question}\n{choice_lines}\nA:"


@torch.inference_mode()
def eval_csqa(model, tokenizer, dataset, batch_size, device, thought_temperature=0.0):
	"""Custom CSQA eval matching the reference impl's letter-token format.

	Left-pads per batch and computes the model's logit at the position right after
	"A:", restricted to the [A,B,C,D,E] token IDs. Compatible with the user's
	`inference_forward` which only returns the last position's logits, because with
	left-padding the last position is each example's actual end of input.
	"""
	letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D", "E"]]
	letter_ids_t = torch.tensor(letter_ids, device=device)

	# We construct prompts manually with left-padding regardless of the tokenizer's
	# default padding_side, by temporarily flipping it and restoring on exit.
	original_side = tokenizer.padding_side
	tokenizer.padding_side = "left"
	try:
		correct = 0
		total = 0
		n = len(dataset)
		for start in range(0, n, batch_size):
			batch = dataset[start:start + batch_size]
			prompts = [
				_format_csqa_prompt(q, c)
				for q, c in zip(batch["question"], batch["choices"])
			]
			labels = [ord(k) - ord("A") for k in batch["answerKey"]]

			encoded = tokenizer(prompts, return_tensors="pt", padding=True)
			input_ids = encoded.input_ids.to(device)
			attention_mask = encoded.attention_mask.to(device)
			# Zero-indexed positions per example: pad → 0 (clamp), real → 0..real_len-1.
			position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)

			out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				thought_temperature=thought_temperature,
			)
			# inference_forward returns shape (b, 1, vocab); take the last (only) position.
			logits = out.logits[:, -1, :]
			letter_logits = logits.index_select(dim=-1, index=letter_ids_t)
			preds = letter_logits.argmax(dim=-1).tolist()
			for pred, label in zip(preds, labels):
				if pred == label:
					correct += 1
				total += 1
	finally:
		tokenizer.padding_side = original_side

	return correct / total if total > 0 else 0.0
