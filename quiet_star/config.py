import yaml
import torch

DEFAULT_CONFIG = {
	"random_seed": 42,
	"root_prefix": ".",
	"project_name": "quiet-star",

	"base_model": {
		"name": "Qwen/Qwen2.5-0.5B",
		"torch_dtype": "bfloat16",
		"device_map": "auto",
		"tokenizer_padding_side": "left",
	},

	"thought_model": {
		"n_thoughts": 2,
		"thought_depth": 8,
		"look_ahead": 4,
		"thought_temperature": 3.0,
		"reinforce_temperature": 3.0,
		"embedding_scale": 100.0,
		"stt_init_id": "---",
		"ett_init_id": "---",
		"beta_thought": 1.0,
		"beta_stability": 1.0,
		"beta_mixed": 1.0,
		"beta_posterior": 1.0,
		"coef_entropy": 0.0,
		"mixer_init_bias": -5.0,
	},

	"dataset": {
		"name": "open-web-math/open-web-math",
		"n_examples": 160_000,
		"train_snippet_length": 96,
	},

	"training": {
		# Custom keys (popped before passing to TrainingArguments)
		"effective_batch_size": 16,
		"eval_lm_batch_size": 512,

		# TrainingArguments fields (passed through directly)
		"per_device_train_batch_size": 4,
		"learning_rate": 2e-6,
		"max_grad_norm": 1.0,
		"max_steps": 100_000,
		"warmup_steps": 20,
		"weight_decay": 0.001,
		"logging_steps": 100,
		"eval_steps": 100,
		"eval_strategy": "steps",
		"eval_on_start": True,
		"save_steps": 500,
		"auto_find_batch_size": True,
		"label_names": ["labels"],
		"include_inputs_for_metrics": True,
		"log_level": "warning",
		"report_to": "wandb",
	},
}


def resolve_torch_dtype(s):
	"""Convert a string like 'bfloat16' to the corresponding torch dtype."""
	mapping = {
		"float16": torch.float16,
		"float32": torch.float32,
		"float64": torch.float64,
		"bfloat16": torch.bfloat16,
		"int8": torch.int8,
		"int16": torch.int16,
		"int32": torch.int32,
		"int64": torch.int64,
	}
	if s in mapping:
		return mapping[s]
	raise ValueError(f"Unknown torch dtype: {s!r}")


def save_config(config, path):
	"""Dump a config dict to a YAML file."""
	with open(path, "w") as f:
		yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(path):
	"""Load a config dict from a YAML file."""
	with open(path, "r") as f:
		return yaml.safe_load(f)
