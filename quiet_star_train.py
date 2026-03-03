import copy
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from functools import partial

from quiet_star.config import DEFAULT_CONFIG, resolve_torch_dtype, save_config, load_config


def config_from_wandb_checkpoint( checkpoint_ref ):
	"""Download the config artifact from the run that produced a checkpoint."""
	import wandb
	api = wandb.Api()
	checkpoint = api.artifact( checkpoint_ref )
	run = checkpoint.logged_by()
	for art in run.logged_artifacts():
		if art.type == "config":
			config_dir = art.download()
			return load_config( os.path.join( config_dir, "config.yaml" ) )
	raise ValueError( f"No config artifact found for run {run.name}" )


SLURM_TEMPLATE = """\
# Slurm job configuration for quiet-star training.
# Keys map directly to sbatch options (simple-slurm).
# See: https://slurm.schedmd.com/sbatch.html

job_name: qstar
partition: gpu
nodes: 1
ntasks_per_node: 1
gres: "gpu:1"
time: "24:00:00"
mem: "32G"
cpus_per_task: 4

# Optional bash commands to run before training (e.g. SSH tunnels, module loads).
# setup: |
#   ssh -fN -D 1080 proxy-host
#   export HTTPS_PROXY=socks5://localhost:1080
"""


def submit_slurm( slurm_config_path, train_args ):
	"""Create a git worktree from HEAD and submit a Slurm job."""
	import yaml
	from simple_slurm import Slurm

	# Ensure working tree is clean
	result = subprocess.run(
		[ "git", "status", "--porcelain", "-uno" ],
		capture_output = True, text = True, check = True
	)
	if result.stdout.strip():
		print( "Error: uncommitted changes. Commit before submitting.", file = sys.stderr )
		sys.exit( 1 )

	# Create worktree from HEAD
	repo_root = subprocess.run(
		[ "git", "rev-parse", "--show-toplevel" ],
		capture_output = True, text = True, check = True
	).stdout.strip()

	ts = int( time.time() )
	worktree_path = os.path.join( repo_root, ".worktrees", f"job-{ts}" )
	os.makedirs( os.path.dirname( worktree_path ), exist_ok = True )

	head_commit = subprocess.run(
		[ "git", "rev-parse", "HEAD" ],
		capture_output = True, text = True, check = True
	).stdout.strip()

	subprocess.run(
		[ "git", "worktree", "add", "--detach", worktree_path, head_commit ],
		check = True
	)

	# Load slurm config and build Slurm object
	with open( slurm_config_path ) as f:
		slurm_params = yaml.safe_load( f )
	setup_script = slurm_params.pop( "setup", None )
	slurm_params.setdefault( "job_name", "qstar" )
	slurm = Slurm( **slurm_params )

	# Copy training config into worktree if provided
	if train_args.path is not None:
		config_basename = os.path.basename( train_args.path )
		shutil.copy2( train_args.path, os.path.join( worktree_path, config_basename ) )

	# Copy DeepSpeed config into worktree if referenced
	if train_args.path is not None:
		cfg = load_config( os.path.join( worktree_path, config_basename ) )
	else:
		cfg = copy.deepcopy( DEFAULT_CONFIG )

	ds_config = cfg.get( "training", {} ).get( "deepspeed" )
	if ds_config is not None and isinstance( ds_config, str ) and os.path.isfile( ds_config ):
		shutil.copy2( ds_config, os.path.join( worktree_path, os.path.basename( ds_config ) ) )

	# Build the train command
	python = sys.executable
	script = os.path.join( worktree_path, "quiet_star_train.py" )
	script_parts = [ script, "train" ]
	if train_args.path is not None:
		script_parts.append( config_basename )
	if train_args.resume is not None:
		script_parts.extend( [ "--resume", train_args.resume ] )
	script_args = " ".join( shlex.quote( p ) for p in script_parts )

	train_cmd = (
		f"srun --kill-on-bad-exit=1"
		f" {shlex.quote( python )} -m torch.distributed.run"
		f" --nnodes=$SLURM_NNODES"
		f" --nproc_per_node=1"
		f" --rdzv_id=$SLURM_JOB_ID"
		f" --rdzv_backend=c10d"
		f" --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
		f" {script_args}" )

	# Setup: cd into worktree; cleanup worktree on exit (only if job succeeded)
	slurm.add_cmd( f"cd {shlex.quote( worktree_path )}" )
	slurm.add_cmd(
		f"trap 'rc=$?;"
		f" kill $(jobs -p) 2>/dev/null;"
		f" if [ $rc -eq 0 ]; then"
		f"  git -C {shlex.quote( repo_root )} worktree remove {shlex.quote( worktree_path )} --force;"
		f" else"
		f'  echo "Job failed (exit $rc); preserving worktree: {worktree_path}" >&2;'
		f" fi' EXIT"
	)

	# Run user-provided setup script (e.g. SSH tunnels, module loads)
	if setup_script is not None:
		slurm.add_cmd( setup_script )

	# Set up distributed env vars
	slurm.add_cmd(
		'export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)' )
	slurm.add_cmd( 'export MASTER_PORT=${MASTER_PORT:-29500}' )

	# Submit
	job_id = slurm.sbatch( train_cmd )
	print( f"Submitted Slurm job {job_id}" )
	print( f"Worktree: {worktree_path}" )
	print( f"Commit: {head_commit[ :8 ]}" )


_interrupted = False


def _graceful_exit_handler( signum, frame ):
	"""On first interrupt, set flag and switch to default handler so second interrupt force-quits."""
	global _interrupted
	signal.signal( signal.SIGINT, signal.SIG_DFL )
	signal.signal( signal.SIGTERM, signal.SIG_DFL )
	_interrupted = True
	name = signal.Signals( signum ).name
	print( f"\n{name} received. Will stop after current step...", file = sys.stderr )
	print( "(Press Ctrl+C again to force quit)", file = sys.stderr )


def train(config, resume_from = None):
	import contextlib
	import random

	import torch
	import transformers
	from lm_eval import simple_evaluate
	from lm_eval.models.huggingface import HFLM
	from lm_eval.tasks import TaskManager
	from torch import nn
	from torch.utils.data import Dataset, IterableDataset

	from quiet_star import ThoughtModelConfig
	from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DynamicCache, PreTrainedModel, \
		DataCollator, PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin, \
		EvalPrediction, TrainerCallback, TrainingArguments, Trainer
	from datasets import load_dataset
	from quiet_star.eval_helpers import preprocess_function, compute_metrics

	torch.backends.cuda.matmul.allow_tf32 = True

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

		def _select_cont_toks( self, logits, contlen = None, inplen = None ):
			# The default implementation of HFLM assumes logits are emitted for
			# the entire input sequence, and truncates them away.
			# That's expensive and unnecessary for our model, but not
			# conforming to it breaks the implementation. However, making this
			# truncation function a no-op is a sufficient fix.
			return logits

	class TrainerWithCache( Trainer ):

		def __init__( self, *args, eval_lm_batch_size = 512, **kwargs ):
			super().__init__( *args, **kwargs )
			self.task_manager = TaskManager()
			self.eval_lm_batch_size = eval_lm_batch_size

		def evaluate(
				self,
				eval_dataset = None,
				ignore_keys = None,
				metric_key_prefix: str = "eval" ):
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

	class ConfigArtifactCallback( TrainerCallback ):
		def __init__( self, config ):
			self.config = config

		def on_train_begin( self, args, state, control, **kwargs ):
			import wandb
			if wandb.run is None or not state.is_world_process_zero:
				return
			config_path = os.path.join( args.output_dir, "config.yaml" )
			save_config( self.config, config_path )
			artifact = wandb.Artifact( name = f"config-{wandb.run.name}", type = "config" )
			artifact.add_file( config_path )
			wandb.run.log_artifact( artifact )

	class GracefulStopCallback( TrainerCallback ):
		def on_step_end( self, args, state, control, **kwargs ):
			if _interrupted:
				print( "Stopping training gracefully...", file = sys.stderr )
				control.should_training_stop = True
			return control

	class SyncEmbeddingsCallback( TrainerCallback ):
		def on_step_end( self, args, state, control, model = None, **kwargs ):
			if model is not None:
				m = model.module if hasattr( model, "module" ) else model
				m.sync_thought_embeddings()
			return control

	signal.signal( signal.SIGINT, _graceful_exit_handler )
	signal.signal( signal.SIGTERM, _graceful_exit_handler )

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
		callbacks = [ ConfigArtifactCallback( config ), GracefulStopCallback(), SyncEmbeddingsCallback() ],
	)

	resume_checkpoint = None
	if resume_from is not None:
		import wandb
		api = wandb.Api()
		artifact = api.artifact( resume_from )
		resume_checkpoint = artifact.download()

	trainer.train( resume_from_checkpoint = resume_checkpoint )


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser( description = "Quiet-Star training" )
	subparsers = parser.add_subparsers( dest = "command", required = True )

	init_parser = subparsers.add_parser( "init", help = "Create a default config file" )
	init_parser.add_argument( "path", nargs = "?", default = "config.yaml", help = "Output path (default: config.yaml)" )

	init_slurm_parser = subparsers.add_parser( "init-slurm", help = "Create a template Slurm config file" )
	init_slurm_parser.add_argument( "path", nargs = "?", default = "slurm.yaml", help = "Output path (default: slurm.yaml)" )

	train_parser = subparsers.add_parser( "train", help = "Train from a config file" )
	train_parser.add_argument( "path", nargs = "?", default = None, help = "Config path (uses built-in defaults if omitted)" )
	train_parser.add_argument( "--resume", metavar = "ARTIFACT", default = None,
		help = "W&B checkpoint artifact to resume from" )
	train_parser.add_argument( "--slurm", metavar = "SLURM_CONFIG", default = None,
		help = "Submit to Slurm using this config file instead of running locally" )

	args = parser.parse_args()

	if args.command == "init":
		save_config( DEFAULT_CONFIG, args.path )
		print( f"Wrote default config to {args.path}" )
	elif args.command == "init-slurm":
		with open( args.path, "w" ) as f:
			f.write( SLURM_TEMPLATE )
		print( f"Wrote Slurm template to {args.path}" )
	elif args.command == "train":
		if args.slurm is not None:
			submit_slurm( args.slurm, args )
		else:
			if args.path is not None:
				cfg = load_config( args.path )
			elif args.resume is not None:
				cfg = config_from_wandb_checkpoint( args.resume )
			else:
				cfg = copy.deepcopy( DEFAULT_CONFIG )
			train( cfg, resume_from = args.resume )
