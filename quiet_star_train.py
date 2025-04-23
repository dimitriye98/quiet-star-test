import contextlib
import copy
from datetime import datetime
from typing import Optional, Union, Callable, Any

import torch
import transformers
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from peft import PeftConfig, get_peft_model, LoraConfig
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from quiet_star import ThoughtModelConfig

torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DynamicCache, PreTrainedModel, DataCollator, \
	PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin, EvalPrediction, TrainerCallback
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import os
import time
from quiet_star.eval_helpers import preprocess_function, compute_metrics

random_seed = 42
torch.manual_seed( random_seed )
random.seed( random_seed )

# MAIN SETUP
root_prefix = "."
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'
# dataset_name = 'c4'
project_name = "quiet-star"
os.environ[ "WANDB_PROJECT" ] = project_name + "-" + dataset_name.split( "/" )[ -1 ]
os.environ[ "WANDB_CACHE_DIR" ] = wandb_cache_dir
os.environ[ "WANDB_LOG_MODEL" ] = "checkpoint"
n_examples = 1_000
full_batch_size = 8
eval_and_logging_steps = 10
save_steps = 100
eval_batch_size = 512

default_params = {
	# "base_model": "mistralai/Mistral-7B-v0.1",
	"base_model": "Qwen/Qwen2.5-0.5B",
	"torch_dtype": torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
	"base_model_kwargs": {
		"torch_dtype": torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
		"device_map": "auto",
		# "cache_dir": root_prefix + "cache",
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
		exclude_modules = ".*(mixer_head|lm_head).*",
		trainable_token_indices = [], # Empty list is signal value for ["<thought>", "</thought>"]
		# modules_to_save = [m for m, _ in test_model.named_modules() if "mixer_head.mlp." in m],
		modules_to_save = [ "mixer_head" ],
	),
	"n_thoughts": 2,
	"thought_depth": 12,
	"look_ahead": 4,
	"look_ahead_pass": 1,
}

tokenizer_args = default_params.pop( "tokenizer_args", [ ] )
tokenizer_kwargs = default_params.pop( "tokenizer_kwargs", { } )
tokenizer = AutoTokenizer.from_pretrained( default_params[ "base_model" ], *tokenizer_args, **tokenizer_kwargs )


def model_init( p ):
	original = False
	params = copy.deepcopy( default_params )
	if p is not None:
		params |= p

	print( params )
	base_model = params.pop( "base_model" )
	base_model_args = params.pop( "base_model_args", [ ] )
	base_model_kwargs = params.pop( "base_model_kwargs", { } )
	peft_config = params.pop( "peft_config", None )

	print( "Loading model" )

	lm_model = AutoModelForCausalLM.from_pretrained(
		base_model,
		*base_model_args,
		**base_model_kwargs
	)

	print( "Loaded model" )
	# tokenizer.pad_token_id = tokenizer.eos_token_id
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	params[ "pad_token_id" ] = tokenizer.pad_token_id

	special_tokens_to_add = [ "<thought>", "</thought>" ]
	tokenizer.add_special_tokens( { "additional_special_tokens": special_tokens_to_add } )
	lm_model.resize_token_embeddings( len( tokenizer ) )

	params[ "start_thought_token_id" ] = tokenizer.convert_tokens_to_ids( "<thought>" )
	params[ "end_thought_token_id" ] = tokenizer.convert_tokens_to_ids( "</thought>" )
	params[ "text_config" ] = lm_model.config

	model_config = ThoughtModelConfig( **params )
	model = AutoModel.from_config( model_config, lm_model = lm_model )

	if peft_config is not None:
		if isinstance( peft_config, dict ):
			peft_config = PeftConfig.from_peft_type( **peft_config )

		# Empty list is magic value
		if hasattr( peft_config, "trainable_token_indices" ) and isinstance(
				peft_config.trainable_token_indices, list ) and not peft_config.trainable_token_indices:
			peft_config.trainable_token_indices = [ params[ "start_thought_token_id" ],
				params[ "end_thought_token_id" ] ]

		model = get_peft_model( model, peft_config )

	model.train()
	return model


# Load dataset
print( "Loading datasets" )
dataset = load_dataset(
	dataset_name,
	"en" if "c4" in dataset_name else "default",
	split = f"train[:{n_examples}]",
	# ignore_verifications=True,
	num_proc = 1,
	# cache_dir=root_prefix + "cache/datasets/",
)

train_dataset = dataset.shuffle( seed = random_seed ).map(
	preprocess_function, batched = True, writer_batch_size = 200 )
# eval_dataset_gsm = load_dataset("gsm8k", "main", split="test").map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200)
# eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split="validation").map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200)
print( "Loaded datasets" )

# eval_datasets = {
# 	"gsm8k": eval_dataset_gsm,
# 	"csqa": eval_dataset_csqa,
# }

batch_size = full_batch_size // default_params[ "n_thoughts" ]
global_gradient_accumulation_steps = full_batch_size // batch_size
ts = int( time.time() )
timestamp = datetime.fromtimestamp( ts )
training_args = TrainingArguments(
	output_dir = root_prefix + f"cache/quietstar/{ts}",
	report_to = "wandb",
	learning_rate = 1e-6,
	optim = "adamw_torch_fused" if torch.cuda.is_available() or torch.backends.mps.is_available() else "adamw_torch",
	per_device_train_batch_size = batch_size,
	per_device_eval_batch_size = batch_size,
	gradient_accumulation_steps = global_gradient_accumulation_steps,
	max_grad_norm = 1.0,
	max_steps = 100000,
	warmup_steps = 20,
	auto_find_batch_size = True,
	weight_decay = 0.001,
	label_names = [ "labels" ],
	include_inputs_for_metrics = True,
	logging_steps = eval_and_logging_steps,
	eval_on_start = True,
	eval_steps = eval_and_logging_steps,
	eval_strategy = "steps",
	save_steps = save_steps,
	run_name = f"n{default_params[ 'n_thoughts' ]}_d{default_params[ "thought_depth" ]}_la{default_params[ "look_ahead" ]}_{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}_{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}",
	# use_mps_device = True,
)


@contextlib.contextmanager
def cm_memory():
	torch.cuda.memory._record_memory_history( max_entries = 100000, stacks = "python" )
	try:
		yield
	finally:
		torch.cuda.memory._dump_snapshot( "snapshot.pickle" )
	torch.cuda.memory._record_memory_history( enabled = None )


class HFLMThought( HFLM ):

	def _model_call( self, inps, attn_mask = None, labels = None ):
		assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
		with torch.inference_mode():
			# Greedy decoding for thoughts during evaluation
			return self.model(inps, thought_temperature = 0.0 ).logits

	def _select_cont_toks( self, logits: torch.Tensor, contlen: int = None, inplen: int = None ) -> torch.Tensor:
		# The default implementation of HFLM assumes logits are emitted for
		# the entire input sequence, and truncates them away.
		# That's expensive and unnecessary for our model, but not
		# conforming to it breaks the implementation. However, making this
		# truncation function a no-op is a sufficient fix.
		return logits


class TrainerWithCache( Trainer ):

	def __init__(
			self, model: Union[ PreTrainedModel, nn.Module, None ] = None, args: TrainingArguments = None,
			data_collator: Optional[ DataCollator ] = None,
			train_dataset: Optional[ Union[ Dataset, IterableDataset, "datasets.Dataset" ] ] = None,
			eval_dataset: Optional[ Union[ Dataset, dict[ str, Dataset ], "datasets.Dataset" ] ] = None,
			processing_class: Optional[
				Union[ PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin ]
			] = None, model_init: Optional[ Callable[ [ ], PreTrainedModel ] ] = None,
			compute_loss_func: Optional[ Callable ] = None,
			compute_metrics: Optional[ Callable[ [ EvalPrediction ], dict ] ] = None,
			callbacks: Optional[ list[ TrainerCallback ] ] = None,
			optimizers: tuple[ Optional[ torch.optim.Optimizer ], Optional[ torch.optim.lr_scheduler.LambdaLR ] ] = (
					None, None),
			optimizer_cls_and_kwargs: Optional[ tuple[ type[ torch.optim.Optimizer ], dict[ str, Any ] ] ] = None,
			preprocess_logits_for_metrics: Optional[
				Callable[ [ torch.Tensor, torch.Tensor ], torch.Tensor ] ] = None ):
		super().__init__(
			model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func,
			compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics )

		self.task_manager = TaskManager()

	def evaluate(
			self,
			eval_dataset = None,
			ignore_keys = None,
			metric_key_prefix: str = "eval" ) -> dict[ str, float ]:
		metric_key_prefix = "eval_"
		override = eval_dataset is not None
		eval_dataset = eval_dataset if override else self.eval_dataset

		hflm = HFLMThought(
			self.model, backend = "causal", batch_size = 512, max_batch_size = 512, tokenizer = self.processing_class )

		results = simple_evaluate(
			model = hflm,
			tasks = eval_dataset,
			task_manager = self.task_manager,
		)
		results = { metric_key_prefix + k: v for k, v in results[ "results" ].items() }

		self.log( results )

		return results

	def compute_loss( self, model, inputs, return_outputs = False, num_items_in_batch = None ):
		inputs[ "past_key_values" ] = DynamicCache()

		with cm_memory() if torch.cuda.is_available() else contextlib.nullcontext():
			loss, o = super().compute_loss(
				model, inputs, return_outputs = True, num_items_in_batch = num_items_in_batch )
		self.log( o[ 2 ] )
		return (loss, o) if return_outputs else loss


trainer = TrainerWithCache(
	args = training_args,
	train_dataset = train_dataset,
	eval_dataset = [ "commonsense_qa" ],  # "gsm8k"],
	compute_metrics = compute_metrics,
	model_init = model_init,
	processing_class = tokenizer,
)

trainer.train()
