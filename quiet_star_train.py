import contextlib
import copy

import torch
from peft import PeftConfig, get_peft_model, LoraConfig

from quiet_star import ThoughtModelConfig, ThoughtModel

torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import os
import time
from quiet_star.eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# MAIN SETUP
root_prefix = "."
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'
# dataset_name = 'c4'
project_name = "quiet-star"
os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
n_ahead_talk_global = 4
n_passes_global = 8
n_ahead_global = 12
n_examples = 1_000
full_batch_size = 8
eval_and_logging_steps = 10
save_steps = 100

default_params = {
	# "base_model": "mistralai/Mistral-7B-v0.1",
	"base_model": "Qwen/Qwen2.5-0.5B",
	"torch_dtype": torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
	"base_model_kwargs": {
		"torch_dtype": torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
		"device_map": "auto",
		#"cache_dir": root_prefix + "cache",
	},
	"tokenizer_kwargs": {
		"padding_side": "right",
	},
	"peft_config": LoraConfig(
		r = 32,
		lora_alpha = 64,
		lora_dropout = 0.1,
		use_rslora = True,
		target_modules = "all-linear",
		exclude_modules = ".*mixer_head.*",
		trainable_token_indices = [], # Empty list is signal value for ["<thought>", "</thought>"]
		# modules_to_save = [m for m, _ in test_model.named_modules() if "mixer_head.mlp." in m],
		modules_to_save = ["mixer_head"],
	),
	"look_ahead_pass": 1,
}

def model_init(p):
	original = False
	params = copy.deepcopy(default_params)
	if p is not None:
		params |= p

	print(params)
	base_model = params.pop("base_model")
	base_model_args = params.pop("base_model_args", [])
	base_model_kwargs = params.pop("base_model_kwargs", {})
	tokenizer_args = params.pop("tokenizer_args", [])
	tokenizer_kwargs = params.pop("tokenizer_kwargs", {})
	peft_config = params.pop("peft_config", None)

	print("Loading model")

	lm_model = AutoModelForCausalLM.from_pretrained(
		base_model,
		*base_model_args,
		**base_model_kwargs
	)

	print("Loaded model")
	tokenizer = AutoTokenizer.from_pretrained(base_model, *tokenizer_args, **tokenizer_kwargs)
	# tokenizer.pad_token_id = tokenizer.eos_token_id
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	params["pad_token_id"] = tokenizer.pad_token_id

	special_tokens_to_add = ["<thought>", "</thought>"]
	tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
	lm_model.resize_token_embeddings(len(tokenizer))

	params["start_thought_token_id"] = tokenizer.convert_tokens_to_ids("<thought>")
	params["end_thought_token_id"] = tokenizer.convert_tokens_to_ids("</thought>")
	params["lm_config"] = lm_model.config

	model_config = ThoughtModelConfig(**params)
	model = ThoughtModel(model_config, lm_model = lm_model)

	if peft_config is not None:
		if isinstance(peft_config, dict):
			peft_config = PeftConfig.from_peft_type(**peft_config)

		# Empty list is magic value
		if hasattr(peft_config, "trainable_token_indices") and isinstance(peft_config.trainable_token_indices, list) and not peft_config.trainable_token_indices:
			peft_config.trainable_token_indices = [params["start_thought_token_id"], params["end_thought_token_id"]]

		model = get_peft_model(model, peft_config)

	model.train()
	return model

# Load dataset
print("Loading datasets")
dataset = load_dataset(
	dataset_name,
	"en" if "c4" in dataset_name else "default",
	split=f"train[:{n_examples}]",
	# ignore_verifications=True,
	num_proc=1,
	#cache_dir=root_prefix + "cache/datasets/",
)

train_dataset = dataset.shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200)
eval_dataset_gsm = load_dataset("gsm8k", "main", split="test").map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200)
eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split="validation").map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200)
print("Loaded datasets")

eval_datasets = {
	"gsm8k": eval_dataset_gsm,
	"csqa": eval_dataset_csqa,
}

batch_size = full_batch_size // n_passes_global
global_gradient_accumulation_steps = full_batch_size // batch_size
run_id = int(time.time())
training_args = TrainingArguments(
	output_dir=root_prefix + f"cache/quietstar/{run_id}",
	# report_to = "wandb",
	learning_rate=1e-6,
	optim="adamw_torch_fused" if torch.cuda.is_available() or torch.backends.mps.is_available() else "adamw_torch",
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	gradient_accumulation_steps=global_gradient_accumulation_steps,
	max_grad_norm=1.0,
	max_steps=100000,
	warmup_steps=20,
	auto_find_batch_size=True,
	weight_decay=0.001,
	label_names=["labels"],
	include_inputs_for_metrics=True,
	logging_steps=eval_and_logging_steps,
	eval_steps=eval_and_logging_steps,
	eval_strategy="steps",
	save_steps=save_steps,
	run_name=f"n={n_ahead_global}_nt={n_ahead_talk_global}_np={n_passes_global}",
	#use_mps_device = True,
)

@contextlib.contextmanager
def cm_memory():
	torch.cuda.memory._record_memory_history(max_entries=100000, stacks="python")
	try:
		yield
	finally:
		torch.cuda.memory._dump_snapshot("snapshot.pickle")
	torch.cuda.memory._record_memory_history(enabled=None)

class TrainerWithCache(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		inputs["past_key_values"] = DynamicCache()

		with cm_memory() if torch.cuda.is_available() else contextlib.nullcontext():
			loss, o = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
		self.log(o[2])
		return (loss, o) if return_outputs else loss


trainer = TrainerWithCache(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    compute_metrics=compute_metrics,
    model_init=model_init,
)

trainer.train()
