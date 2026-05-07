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
def eval_csqa(model, tokenizer, dataset, batch_size, device, thought_temperature=0.0, debug_first_batch=False):
	"""Custom CSQA eval matching the reference impl's letter-token format.

	Left-pads per batch and reads the model's logit at the position right after
	"A:". Reports two metrics:
	- ``acc``: argmax over [A,B,C,D,E] (exact-match accuracy).
	- ``soft_acc``: average renormalized probability of the correct letter under
	  softmax over [A,B,C,D,E,\\n] — this is the metric the reference impl's
	  ``compute_metrics`` returns and what the paper's reported numbers reflect.

	If ``debug_first_batch`` is True, prints the top-10 predicted tokens (over
	the full vocabulary) at the answer position for the first example of the
	first batch. Useful for diagnosing whether the model is actually looking at
	the right position — top tokens should at least include letters A-E or
	plausible structural tokens (newline, space) for a sane Mistral baseline.

	Compatible with our ``inference_forward`` returning only the last position's
	logits, because with left-padding the last position is each example's actual
	end of input.
	"""
	# A/B/C/D/E ids for Mistral. Newline is included in the soft-acc denominator
	# to match the reference's `valid_letter_tokens = [330, 365, 334, 384, 413, 13]`.
	letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D", "E"]]
	letter_ids_t = torch.tensor(letter_ids, device=device)
	soft_ids_t = torch.tensor(letter_ids + [13], device=device)

	original_side = tokenizer.padding_side
	tokenizer.padding_side = "left"
	try:
		correct = 0
		soft_sum = 0.0
		total = 0
		first_batch = True
		n = len(dataset)
		for start in range(0, n, batch_size):
			batch = dataset[start:start + batch_size]
			prompts = [
				_format_csqa_prompt(q, c)
				for q, c in zip(batch["question"], batch["choices"])
			]
			labels = torch.tensor([ord(k) - ord("A") for k in batch["answerKey"]], device=device)

			encoded = tokenizer(prompts, return_tensors="pt", padding=True)
			input_ids = encoded.input_ids.to(device)
			attention_mask = encoded.attention_mask.to(device)
			position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)

			out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				thought_temperature=thought_temperature,
			)
			logits = out.logits[:, -1, :].float()

			if debug_first_batch and first_batch:
				first_batch = False
				print(f"[eval_csqa debug] prompt:\n{prompts[0]!r}", flush=True)
				print(f"[eval_csqa debug] correct answer: {batch['answerKey'][0]}", flush=True)

				# Weight-level sanity: pretrained Mistral lm_head/embed_tokens have
				# specific statistics (std ~0.04, non-uniform per-row norms because
				# of training). Uninitialized weights are uniform random, std ~0.02.
				lh = model.lm_model.lm_head.weight.detach().float()
				et = model.lm_model.model.embed_tokens.weight.detach().float()
				print(f"[eval_csqa debug:weights] lm_head shape={tuple(lh.shape)} dtype={lh.dtype} mean={lh.mean().item():.5f} std={lh.std().item():.5f}", flush=True)
				print(f"[eval_csqa debug:weights] embed_tokens shape={tuple(et.shape)} dtype={et.dtype} mean={et.mean().item():.5f} std={et.std().item():.5f}", flush=True)
				# Per-row norm distribution: pretrained models have a wide range,
				# uninitialized weights have all rows ~equal norm.
				lh_norms = lh.norm(dim=-1)
				et_norms = et.norm(dim=-1)
				print(f"[eval_csqa debug:weights] lm_head per-row norm: min={lh_norms.min().item():.4f} max={lh_norms.max().item():.4f} std={lh_norms.std().item():.4f}", flush=True)
				print(f"[eval_csqa debug:weights] embed_tokens per-row norm: min={et_norms.min().item():.4f} max={et_norms.max().item():.4f} std={et_norms.std().item():.4f}", flush=True)

				def _print_top(label, lg):
					p = lg.softmax(dim=-1)
					top_p, top_i = p.topk(10)
					print(f"[eval_csqa debug:{label}] top-10 next-token preds:", flush=True)
					for pp, ii in zip(top_p.tolist(), top_i.tolist()):
						print(f"  {ii:6d} {tokenizer.decode([ii])!r:>16s}  {pp:.4f}", flush=True)
					letter_p = p.gather(0, letter_ids_t).tolist()
					print(f"[eval_csqa debug:{label}] P(A)={letter_p[0]:.4f} P(B)={letter_p[1]:.4f} P(C)={letter_p[2]:.4f} P(D)={letter_p[3]:.4f} P(E)={letter_p[4]:.4f}", flush=True)

				# Sanity check: a simple prompt that any working Mistral-7B should
				# continue with " John" / " Sarah" / etc. If this gives garbage too,
				# the model load is broken (Liger, dtype, weights, or wrong model id).
				sanity_prompt = "Hello, my name is"
				sanity_ids = tokenizer(sanity_prompt, return_tensors="pt").input_ids.to(device)
				sanity_out = model.lm_model(
					input_ids=sanity_ids,
					use_cache=False,
					output_hidden_states=False,
					return_dict=True,
					logits_to_keep=1,
				)
				sanity_logits = sanity_out.logits[:, -1, :].float()
				print(f"[eval_csqa debug:sanity] prompt: {sanity_prompt!r}", flush=True)
				_print_top("sanity", sanity_logits[0])
				del sanity_out, sanity_logits

				_print_top("inference_forward", logits[0])

				# Slice to just the first example for the bypass calls so we don't
				# triple-forward the full batch and OOM.
				ex_input_ids = input_ids[:1]
				ex_attention_mask = attention_mask[:1]
				ex_position_ids = position_ids[:1]

				# Bypass inference_forward and call the underlying Mistral directly
				# with the same input/mask/position to see if the issue is in our wrapper
				# or in our position_ids / attention_mask setup.
				raw_out = model.lm_model(
					input_ids=ex_input_ids,
					attention_mask=ex_attention_mask,
					position_ids=ex_position_ids,
					use_cache=False,
					output_hidden_states=False,
					return_dict=True,
					logits_to_keep=1,
				)
				raw_logits = raw_out.logits[:, -1, :].float()
				_print_top("lm_model_direct", raw_logits[0])
				del raw_out, raw_logits

				# Also try without position_ids (let HF derive from cache_position).
				raw_out2 = model.lm_model(
					input_ids=ex_input_ids,
					attention_mask=ex_attention_mask,
					use_cache=False,
					output_hidden_states=False,
					return_dict=True,
					logits_to_keep=1,
				)
				raw_logits2 = raw_out2.logits[:, -1, :].float()
				_print_top("lm_model_no_pos_ids", raw_logits2[0])
				del raw_out2, raw_logits2

			# Argmax over the 5 letters
			letter_logits = logits.index_select(dim=-1, index=letter_ids_t)
			preds = letter_logits.argmax(dim=-1)
			correct += (preds == labels).sum().item()

			# Soft accuracy: renormalized prob over [A,B,C,D,E,\n] of the correct letter
			soft_logits = logits.index_select(dim=-1, index=soft_ids_t)
			soft_probs = soft_logits.softmax(dim=-1)
			soft_sum += soft_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1).sum().item()

			total += labels.shape[0]
	finally:
		tokenizer.padding_side = original_side

	if total == 0:
		return {"acc": 0.0, "soft_acc": 0.0}
	return {"acc": correct / total, "soft_acc": soft_sum / total}
