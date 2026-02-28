import contextlib
import copy
from datetime import datetime
from typing import Optional, Union, Callable, Any
from functools import partial

import torch
import transformers
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from quiet_star import ThoughtModelConfig
from quiet_star.config import DEFAULT_CONFIG, resolve_torch_dtype, save_config, load_config

torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DynamicCache, PreTrainedModel, DataCollator, \
	PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin, EvalPrediction, TrainerCallback
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import os
import time
from quiet_star.eval_helpers import preprocess_function, compute_metrics


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
				Callable[ [ torch.Tensor, torch.Tensor ], torch.Tensor ] ] = None,
			eval_lm_batch_size: int = 512 ):
		super().__init__(
			model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func,
			compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics )

		self.task_manager = TaskManager()
		self.eval_lm_batch_size = eval_lm_batch_size

	def evaluate(
			self,
			eval_dataset = None,
			ignore_keys = None,
			metric_key_prefix: str = "eval" ) -> dict[ str, float ]:
		metric_key_prefix = "eval_"
		override = eval_dataset is not None
		eval_dataset = eval_dataset if override else self.eval_dataset

		hflm = HFLMThought(
			self.model, backend = "causal", batch_size = self.eval_lm_batch_size,
			max_batch_size = self.eval_lm_batch_size, tokenizer = self.processing_class )

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


def train(config):
	torch.manual_seed( config["random_seed"] )
	random.seed( config["random_seed"] )

	os.environ[ "WANDB_PROJECT" ] = config["project_name"] + "-" + config["dataset"]["name"].split( "/" )[ -1 ]
	os.environ[ "WANDB_CACHE_DIR" ] = config["root_prefix"] + "cache/quietstar/wandb_cache"
	os.environ[ "WANDB_LOG_MODEL" ] = "checkpoint"

	tokenizer = AutoTokenizer.from_pretrained(
		config["base_model"]["name"],
		padding_side = config["base_model"]["tokenizer_padding_side"],
	)

	def model_init( p ):
		params = copy.deepcopy( config["thought_model"] )
		if p is not None:
			params |= p

		print( params )

		dtype = resolve_torch_dtype( config["base_model"]["torch_dtype"] )

		print( "Loading model" )

		lm_model = AutoModelForCausalLM.from_pretrained(
			config["base_model"]["name"],
			torch_dtype = dtype,
			device_map = config["base_model"]["device_map"],
		)

		print( "Loaded model" )
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token_id = tokenizer.eos_token_id

		params[ "pad_token_id" ] = tokenizer.pad_token_id

		special_tokens_to_add = [ "<thought>", "</thought>" ]
		tokenizer.add_special_tokens( { "additional_special_tokens": special_tokens_to_add } )
		lm_model.resize_token_embeddings( len( tokenizer ) )

		stt_init_id = params.pop( "stt_init_id", None )
		ett_init_id = params.pop( "ett_init_id", None )

		params["stt_init_id"] = tokenizer.convert_tokens_to_ids( stt_init_id )
		params["ett_init_id"] = tokenizer.convert_tokens_to_ids( ett_init_id )

		params[ "start_thought_token_id" ] = tokenizer.convert_tokens_to_ids( "<thought>" )
		params[ "end_thought_token_id" ] = tokenizer.convert_tokens_to_ids( "</thought>" )
		params[ "text_config" ] = lm_model.config

		model_config = ThoughtModelConfig( **params )
		model = AutoModel.from_config( model_config, lm_model = lm_model )

		model.train()
		return model

	# Load dataset
	print( "Loading datasets" )
	ds_cfg = config["dataset"]
	dataset = load_dataset(
		ds_cfg["name"],
		"en" if "c4" in ds_cfg["name"] else "default",
		split = f"train[:{ds_cfg['n_examples']}]",
		num_proc = 1,
	)

	train_dataset = dataset.shuffle( seed = config["random_seed"] ).map(
		partial( preprocess_function, max_length = ds_cfg["train_snippet_length"] ), batched = True, writer_batch_size = 200 )
	print( "Loaded datasets" )

	tr = copy.deepcopy( config["training"] )

	# Pop custom keys that aren't TrainingArguments fields
	effective_batch_size = tr.pop( "effective_batch_size" )
	eval_lm_batch_size = tr.pop( "eval_lm_batch_size" )

	# Compute derived TrainingArguments fields
	ts = int( time.time() )
	timestamp = datetime.fromtimestamp( ts )
	tr["output_dir"] = config["root_prefix"] + f"cache/quietstar/{ts}"
	tr["optim"] = "adamw_torch_fused" if torch.cuda.is_available() or torch.backends.mps.is_available() else "adamw_torch"
	tr["gradient_accumulation_steps"] = effective_batch_size // tr["per_device_train_batch_size"]
	tr["per_device_eval_batch_size"] = tr["per_device_train_batch_size"]
	tr["run_name"] = f"n{config['thought_model']['n_thoughts']}_d{config['thought_model']['thought_depth']}_la{config['thought_model']['look_ahead']}_{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}_{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}"

	training_args = TrainingArguments( **tr )

	trainer = TrainerWithCache(
		args = training_args,
		train_dataset = train_dataset,
		eval_dataset = [ "commonsense_qa" ],
		compute_metrics = compute_metrics,
		model_init = model_init,
		processing_class = tokenizer,
		eval_lm_batch_size = eval_lm_batch_size,
	)

	trainer.train()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser( description = "Quiet-Star training" )
	subparsers = parser.add_subparsers( dest = "command", required = True )

	init_parser = subparsers.add_parser( "init", help = "Create a default config file" )
	init_parser.add_argument( "path", nargs = "?", default = "config.yaml", help = "Output path (default: config.yaml)" )

	train_parser = subparsers.add_parser( "train", help = "Train from a config file" )
	train_parser.add_argument( "path", nargs = "?", default = "config.yaml", help = "Config path (default: config.yaml)" )

	args = parser.parse_args()

	if args.command == "init":
		save_config( DEFAULT_CONFIG, args.path )
		print( f"Wrote default config to {args.path}" )
	elif args.command == "train":
		cfg = load_config( args.path )
		train( cfg )
