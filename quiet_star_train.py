import copy
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from functools import partial

from quiet_star.config import DEFAULT_CONFIG, resolve_torch_dtype, save_config, load_config


_PENDING_MANIFEST = "pending_upload.json"
_AUTO_RESUME_MARKER = ".auto_resume"


def _resolve_wandb_cache_dir( root_prefix_arg, path_cfg ):
	"""Compute the WANDB_CACHE_DIR value relative to the dispatch shell's CWD.
	Returns None if an existing WANDB_CACHE_DIR should be left alone.
	Priority: --root-prefix CLI > existing WANDB_CACHE_DIR > config's root_prefix
	> dispatch CWD. The path is always absolute so it survives env propagation
	to a Slurm job that runs in a different CWD."""
	if "WANDB_CACHE_DIR" in os.environ and root_prefix_arg is None:
		return None
	if root_prefix_arg is not None:
		root_prefix = root_prefix_arg
	elif path_cfg is not None:
		root_prefix = path_cfg.get( "root_prefix", "." )
	else:
		root_prefix = "."
	return os.path.abspath( os.path.join( os.path.expanduser( root_prefix ), "cache/quietstar/wandb_cache" ) )


def _resolve_resume( resume_from ):
	"""Resolve --resume to (checkpoint_artifact, run) and set WANDB_RUN_ID/RESUME
	in env as a side effect — this needs to happen as early as possible (before
	any further wandb operation that might cache settings) so that wandb.init()
	in WandbCallback.setup() resumes the original run rather than starting a new
	one. Accepts either a wandb artifact ref (entity/project/name:version) or a
	run path (entity/project/run_id); for a run path, picks the most recently
	logged mid-training checkpoint (identified by a `checkpoint_global_step_<step>`
	alias, which is what HF's WandbCallback.on_save tags resumable checkpoints
	with — distinct from end-of-train artifacts tagged `last`/`best`, which only
	contain model weights and aren't resumable)."""
	import wandb
	api = wandb.Api()
	if ":" in resume_from.rsplit( "/", 1 )[ -1 ]:
		artifact = api.artifact( resume_from )
		run = artifact.logged_by()
	else:
		run = api.run( resume_from )
		candidates = [
			a for a in run.logged_artifacts()
			if a.type == "model" and any( al.startswith( "checkpoint_global_step_" ) for al in a.aliases )
		]
		if not candidates:
			raise ValueError( f"No resumable mid-training checkpoints logged by run {resume_from} (only end-of-train artifacts found)" )
		artifact = max( candidates, key = lambda a: a.created_at )
	os.environ[ "WANDB_RUN_ID" ] = run.id
	os.environ[ "WANDB_RESUME" ] = "must"
	return artifact, run


def config_from_wandb_checkpoint( resume_from ):
	"""Download the config artifact from the run associated with --resume."""
	print( f"[config_from_wandb] WANDB_CACHE_DIR = {os.environ.get( 'WANDB_CACHE_DIR', '<unset>' )}", file = sys.stderr )
	_, run = _resolve_resume( resume_from )
	for art in run.logged_artifacts():
		if art.type == "config":
			config_dir = art.download()
			return load_config( os.path.join( config_dir, "config.yaml" ) )
	raise ValueError( f"No config artifact found for run {run.name}" )


def _has_checkpoint_subdirs( path ):
	if not os.path.isdir( path ):
		return False
	for entry in os.listdir( path ):
		if entry.startswith( "checkpoint-" ) and os.path.isdir( os.path.join( path, entry ) ):
			try:
				int( entry.split( "-" )[ 1 ] )
				return True
			except ValueError:
				continue
	return False


def _latest_checkpoint( output_dir ):
	candidates = []
	for entry in os.listdir( output_dir ):
		if entry.startswith( "checkpoint-" ) and os.path.isdir( os.path.join( output_dir, entry ) ):
			try:
				candidates.append( ( int( entry.split( "-" )[ 1 ] ), entry ) )
			except ValueError:
				continue
	if not candidates:
		raise ValueError( f"No checkpoint-N subdirs in {output_dir}" )
	return os.path.join( output_dir, max( candidates )[ 1 ] )


def _resolve_local_resume( path ):
	"""Resolve a local --resume path to (checkpoint_dir, output_dir, run_info).
	Accepts:
	- a `checkpoint-N` dir,
	- an `output_dir` (parent of checkpoint-N, has `wandb_run.json`),
	- any ancestor (e.g. a worktree); the most recently modified output_dir
	  found beneath it is selected.
	Recovers the wandb run id from the durable `wandb_run.json` sidecar
	written at training start, falling back to a `pending_upload.json`
	manifest if the sidecar is missing."""
	path = os.path.abspath( path )
	if os.path.exists( os.path.join( path, "trainer_state.json" ) ):
		ckpt_dir = path
		output_dir = os.path.dirname( path )
	elif os.path.exists( os.path.join( path, "wandb_run.json" ) ) or _has_checkpoint_subdirs( path ):
		ckpt_dir = _latest_checkpoint( path )
		output_dir = path
	else:
		# Walk for output_dirs (marked by wandb_run.json + checkpoint-N subdirs).
		output_dirs = []
		for root, dirs, files in os.walk( path ):
			if "wandb_run.json" in files and _has_checkpoint_subdirs( root ):
				output_dirs.append( root )
				dirs[ : ] = []  # don't recurse into output_dirs
		if not output_dirs:
			raise ValueError( f"No resumable training output_dir found under {path}" )
		output_dir = max( output_dirs, key = lambda d: os.path.getmtime( d ) )
		ckpt_dir = _latest_checkpoint( output_dir )

	sidecar = os.path.join( output_dir, "wandb_run.json" )
	if os.path.exists( sidecar ):
		with open( sidecar ) as f:
			run_info = json.load( f )
	else:
		manifest = os.path.join( ckpt_dir, _PENDING_MANIFEST )
		if not os.path.exists( manifest ):
			raise ValueError(
				f"No wandb_run.json or pending_upload.json near {path}; "
				f"cannot recover run id for resume." )
		with open( manifest ) as f:
			m = json.load( f )
		run_info = {
			"run_id": m[ "wandb" ][ "run_id" ],
			"project": m[ "wandb" ][ "project" ],
			"entity": m[ "wandb" ].get( "entity" ),
		}
	return ckpt_dir, output_dir, run_info


def config_from_local_checkpoint( path ):
	"""Load config.yaml from a local checkpoint path's containing output_dir."""
	_, output_dir, _ = _resolve_local_resume( path )
	config_path = os.path.join( output_dir, "config.yaml" )
	if not os.path.exists( config_path ):
		raise ValueError( f"No config.yaml at {config_path}" )
	return load_config( config_path )


def submit_uploader( pending_dir ):
	"""Submit a sibling sbatch job that uploads pending_dir to wandb. Reads
	.uploader_config.json from the worktree (assumed to be the dir containing
	this script). Uses `sbatch --wrap` with a multi-line bash payload (setup
	commands + exec python run-uploader-job) so the uploader has no dependency
	on a script file in the worktree — the trap may have removed the worktree
	by the time the uploader's sbatch job actually runs."""
	cfg_path = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".uploader_config.json" )
	if not os.path.exists( cfg_path ):
		print( f"[uploader] no .uploader_config.json at {cfg_path}; skipping", file = sys.stderr )
		return
	with open( cfg_path ) as f:
		cfg = json.load( f )
	setup = cfg.get( "setup" ) or ""
	setup_block = ( setup.rstrip() + "\n" ) if setup else ""
	wrap = (
		"set -euo pipefail\n"
		+ setup_block
		+ f"exec {shlex.quote( cfg[ 'python' ] )} {shlex.quote( cfg[ 'script' ] )} run-uploader-job\n" )
	# --chdir keeps slurm-%j.out outside the worktree (the training job's trap
	# may eventually remove the worktree on a clean run completion).
	cmd = [
		"sbatch",
		f"--chdir={cfg[ 'repo_root' ]}",
		*cfg[ "sbatch_flags" ],
		"--export=ALL",
		f"--wrap={wrap}",
	]
	env = os.environ.copy()
	env[ "QUIET_STAR_PENDING_DIR" ] = os.path.abspath( pending_dir )
	subprocess.run( cmd, env = env, check = True )


def run_uploader_job():
	"""Runs as the uploader sbatch job. Uploads the pending checkpoint(s) and
	exits. The worktree is intentionally left alone — auto-resume needs it, and
	cleanup only happens when the *training* job exits cleanly (rc=0 trap branch
	in submit_slurm). Reads pending dir from env set via submit_uploader's
	--export=ALL."""
	pending = os.environ[ "QUIET_STAR_PENDING_DIR" ]
	upload_pending( pending )


def upload_pending( output_dir ):
	"""Walk output_dir for `*/pending_upload.json` manifests left by graceful-stop
	handlers and upload each to wandb. The manifest mirrors what HF's
	WandbCallback.on_save would have logged in-process. On success, the manifest
	is removed; the checkpoint dir is preserved so --resume from the local path
	still works. On failure, the manifest stays so a re-run can retry."""
	import wandb

	manifests = []
	if os.path.isfile( os.path.join( output_dir, _PENDING_MANIFEST ) ):
		manifests.append( os.path.join( output_dir, _PENDING_MANIFEST ) )
	for entry in sorted( os.listdir( output_dir ) ):
		path = os.path.join( output_dir, entry, _PENDING_MANIFEST )
		if os.path.isfile( path ):
			manifests.append( path )

	if not manifests:
		print( f"[upload-pending] no manifests under {output_dir}", file = sys.stderr )
		return

	for manifest_path in manifests:
		with open( manifest_path ) as f:
			manifest = json.load( f )
		ckpt_dir = manifest[ "checkpoint_dir" ]
		w = manifest[ "wandb" ]
		print( f"[upload-pending] uploading {ckpt_dir} -> {w['artifact_name']} (run {w['run_id']})", file = sys.stderr )
		wandb.init(
			id = w[ "run_id" ],
			project = w[ "project" ],
			entity = w.get( "entity" ),
			resume = "must",
		)
		artifact = wandb.Artifact(
			name = w[ "artifact_name" ],
			type = w.get( "artifact_type", "model" ),
			metadata = w.get( "metadata", {} ),
		)
		artifact.add_dir( ckpt_dir )
		wandb.log_artifact( artifact, aliases = w.get( "aliases", [] ) )
		wandb.finish()
		os.remove( manifest_path )
		print( f"[upload-pending] uploaded and removed {manifest_path}", file = sys.stderr )


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

# Auto-requeue on Slurm preemption is enabled by default — set `requeue: false`
# to opt out. With requeue, a preempted job is put back in the queue with the
# same JOB_ID; on its next run, output_dir (keyed on $SLURM_JOB_ID) is the same
# and an `.auto_resume` marker (written only on SIGTERM, not SIGINT/Ctrl-C)
# triggers automatic resume from the latest local checkpoint.

# Optional bash commands to run on the head node before training. Env vars
# exported here propagate to all ranks via srun --export=ALL — put module
# loads, http_proxy exports, etc. here.
# setup: |
#   module load nvhpc
#   export http_proxy=socks5://localhost:1080
#   export https_proxy=socks5://localhost:1080

# Optional per-node bash commands, run as part of the training srun's per-task
# script (one execution per allocated node) before torchrun starts. Use this
# for per-node daemons (SOCKS proxies, fuse mounts) that need to outlive setup
# — running them in a separate srun step has them killed by Slurm cgroup
# cleanup the moment that step ends. Env vars exported here DO propagate to
# training because we exec into torchrun in the same shell.
# per_node_setup: |
#   ssh -fN -D 1080 proxy-host

# Optional follow-on job for uploading checkpoints saved on Slurm preemption.
# On graceful preempt the handler saves a checkpoint + manifest locally without
# uploading to wandb (the upload would exceed Slurm's grace period). At the
# next training start, if pending manifests exist in output_dir, an uploader
# sbatch is fired with the overrides below (merged onto the main config; set
# a key to `null` to remove it). Without an `uploader:` section, manifests are
# left in place and must be uploaded manually with
# `quiet_star_train.py upload-pending <output_dir>`.
#
# `setup` is inherited from the top-level setup by default (so SOCKS proxies
# needed for wandb access apply automatically). Set `uploader.setup: null` to
# disable, or supply a different block here to override.
# uploader:
#   partition: cpu
#   gres: null
#   time: "06:00:00"
#   mem: "8G"
#   cpus_per_task: 2
"""


def _worktree_for_path( path, repo_root ):
	"""If path lies inside a worktree under repo_root/.worktrees/, return the
	worktree's top-level dir; otherwise None."""
	abspath = os.path.realpath( os.path.abspath( path ) )
	worktrees_root = os.path.realpath( os.path.join( repo_root, ".worktrees" ) )
	if not abspath.startswith( worktrees_root + os.sep ):
		return None
	rel = os.path.relpath( abspath, worktrees_root )
	return os.path.join( worktrees_root, rel.split( os.sep )[ 0 ] )


def submit_slurm( slurm_config_path, train_args ):
	"""Create (or reuse) a git worktree and submit a Slurm job. If --resume
	points inside an existing `.worktrees/<job-X>/` worktree, that worktree is
	reused (no fresh checkout) and the clean-tree check is skipped — the resume
	state is what matters, not the current state of the user's main checkout."""
	import yaml
	from simple_slurm import Slurm

	repo_root = subprocess.run(
		[ "git", "rev-parse", "--show-toplevel" ],
		capture_output = True, text = True, check = True
	).stdout.strip()

	# Detect existing worktree to reuse, based on --resume path
	existing_worktree = None
	if train_args.resume is not None and os.path.isdir( train_args.resume ):
		existing_worktree = _worktree_for_path( train_args.resume, repo_root )

	if existing_worktree is None:
		# Fresh worktree: require clean tree so the new checkout captures HEAD intent
		result = subprocess.run(
			[ "git", "status", "--porcelain", "-uno" ],
			capture_output = True, text = True, check = True
		)
		if result.stdout.strip():
			print( "Error: uncommitted changes. Commit before submitting.", file = sys.stderr )
			sys.exit( 1 )

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
	else:
		worktree_path = existing_worktree
		head_commit = subprocess.run(
			[ "git", "-C", worktree_path, "rev-parse", "HEAD" ],
			capture_output = True, text = True, check = True
		).stdout.strip()
		print(
			f"[submit_slurm] Reusing worktree {worktree_path} (HEAD={head_commit[ :8 ]})",
			file = sys.stderr )

	# Load slurm config and build Slurm object
	with open( slurm_config_path ) as f:
		slurm_params = yaml.safe_load( f )
	setup_script = slurm_params.pop( "setup", None )
	per_node_setup = slurm_params.pop( "per_node_setup", None )
	uploader_overrides = slurm_params.pop( "uploader", None )
	slurm_params.setdefault( "job_name", "qstar" )
	# Auto-requeue on preemption: Slurm puts the job back in queue with the same
	# JOB_ID and our auto-resume picks up from the latest local checkpoint.
	# User can opt out with `requeue: false` in slurm.yaml.
	slurm_params.setdefault( "requeue", True )
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

	# Resolve WANDB_CACHE_DIR at dispatch (in the local CWD) so root_prefix="."
	# in a config means the user's local checkout, not the per-job worktree.
	# Propagated to the Slurm job via sbatch's default env inheritance.
	cache_dir = _resolve_wandb_cache_dir( train_args.root_prefix, cfg if train_args.path is not None else None )
	if cache_dir is not None:
		os.environ[ "WANDB_CACHE_DIR" ] = cache_dir
	print( f"[submit_slurm] WANDB_CACHE_DIR = {os.environ.get( 'WANDB_CACHE_DIR', '<unset>' )}", file = sys.stderr )

	# Build the train command
	python = sys.executable
	script = os.path.join( worktree_path, "quiet_star_train.py" )
	script_parts = [ script, "train" ]
	if train_args.path is not None:
		script_parts.append( config_basename )
	if train_args.resume is not None:
		# The inner sbatch script `cd`s to the new worktree before launching
		# python, so a relative --resume path resolves against the wrong CWD on
		# the inner side and falls through to the wandb-artifact path with a
		# confusing error. Absolutize local-directory resumes here; leave wandb
		# refs alone.
		resume_arg = train_args.resume
		if os.path.isdir( resume_arg ):
			resume_arg = os.path.abspath( resume_arg )
		script_parts.extend( [ "--resume", resume_arg ] )
	script_args = " ".join( shlex.quote( p ) for p in script_parts )

	torchrun_cmd = (
		f"{shlex.quote( python )} -m torch.distributed.run"
		f" --nnodes=$SLURM_NNODES"
		f" --nproc_per_node=$SLURM_GPUS_ON_NODE"
		f" --rdzv_id=$SLURM_JOB_ID"
		f" --rdzv_backend=c10d"
		f" --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
		f" {script_args}" )
	# per_node_setup runs as part of the training srun's per-task script so its
	# child processes (e.g. backgrounded `ssh -fN -D` SOCKS proxies) stay in the
	# same Slurm cgroup as torchrun and live for the whole training step.
	# Running them in a separate srun step has them killed by Slurm cgroup
	# cleanup the moment that step ends — before training even starts.
	if per_node_setup is not None:
		inner = (
			"set -euo pipefail\n"
			+ per_node_setup.rstrip()
			+ f"\nexec {torchrun_cmd}\n" )
		train_cmd = (
			"srun --export=ALL --kill-on-bad-exit=1 bash -c "
			+ shlex.quote( inner ) )
	else:
		train_cmd = f"srun --export=ALL --kill-on-bad-exit=1 {torchrun_cmd}"

	# If the slurm config has an `uploader:` section, write a sidecar with the
	# merged sbatch flags + setup commands. The training process reads this at
	# startup to fire follow-on uploader jobs for any pending manifests. Without
	# an `uploader:` section, no sidecar is written and pending manifests must
	# be uploaded manually with `quiet_star_train.py upload-pending <output_dir>`.
	#
	# Setup resolution: an explicit `uploader.setup` (including null) wins;
	# otherwise inherit the top-level `setup` so things like SOCKS proxies
	# needed for wandb access apply by default.
	uploader_config_path = os.path.join( worktree_path, ".uploader_config.json" )
	if uploader_overrides is not None:
		if "setup" in uploader_overrides:
			uploader_setup = uploader_overrides.pop( "setup" )
		else:
			uploader_setup = setup_script
		uploader_flags = dict( slurm_params )
		# Uploader is conceptually single-node single-task — the main job's
		# nodes/ntasks-per-node/mem-per-cpu/etc. are sized for distributed
		# training and don't make sense for a one-shot wandb upload. Reset to
		# minimal defaults; the user can override via uploader_overrides.
		uploader_flags[ "nodes" ] = 1
		uploader_flags.pop( "ntasks_per_node", None )
		for k, v in uploader_overrides.items():
			if v is None:
				uploader_flags.pop( k, None )
			else:
				uploader_flags[ k ] = v
		uploader_flags[ "job_name" ] = uploader_flags.get( "job_name", "qstar" ) + "-upload"
		# Uploader is dispatched without a Slurm dependency (it runs concurrent
		# with the training job that fires it), so strip any inherited dependency.
		uploader_flags.pop( "dependency", None )
		# Uploader doesn't requeue (a single one-shot upload, idempotent only via
		# manual retry of upload-pending).
		uploader_flags.pop( "requeue", None )
		flag_strs = []
		for k, v in uploader_flags.items():
			opt = "--" + k.replace( "_", "-" )
			if isinstance( v, bool ):
				if v:
					flag_strs.append( opt )
			else:
				flag_strs.append( f"{opt}={v}" )

		with open( uploader_config_path, "w" ) as f:
			json.dump(
				{
					"sbatch_flags": flag_strs,
					"setup": uploader_setup,
					"python": python,
					"script": script,
					"repo_root": repo_root,
				},
				f, indent = 2 )

	# Trap: rc==0 (training completed cleanly) → remove worktree; otherwise
	# (graceful preempt that left a manifest, real failure, scancel, etc.) →
	# preserve worktree so the next requeue / --resume can pick it up.
	slurm.add_cmd( f"cd {shlex.quote( worktree_path )}" )
	slurm.add_cmd( f"WORKTREE_PATH={shlex.quote( worktree_path )}" )
	slurm.add_cmd( f"REPO_ROOT={shlex.quote( repo_root )}" )
	slurm.add_cmd(
		"trap 'rc=$?;"
		" kill $(jobs -p) 2>/dev/null;"
		" if [ $rc -eq 0 ]; then"
		'  git -C "$REPO_ROOT" worktree remove "$WORKTREE_PATH" --force;'
		" else"
		'  echo "Job exited rc=$rc; preserving worktree: $WORKTREE_PATH" >&2;'
		" fi' EXIT"
	)

	# Run user-provided head-node setup (env vars propagated to all ranks via
	# srun --export=ALL).
	if setup_script is not None:
		slurm.add_cmd( setup_script )

	# (per_node_setup is wired into train_cmd's per-task bash above, so each
	# allocated node runs it before exec'ing torchrun. See the long-form
	# rationale there.)

	# Set up distributed env vars
	slurm.add_cmd(
		'export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)' )
	slurm.add_cmd( 'export MASTER_PORT=${MASTER_PORT:-29500}' )

	# Submit
	dispatch_cwd = os.getcwd()
	job_id = slurm.sbatch( train_cmd )
	print( f"Submitted Slurm job {job_id}" )
	print( f"Worktree: {worktree_path}" )
	print( f"Commit: {head_commit[ :8 ]}" )

	if getattr( train_args, "follow", False ):
		_follow_slurm_log( slurm_params, job_id, dispatch_cwd )


def _follow_slurm_log( slurm_params, job_id, dispatch_cwd ):
	"""Wait for the slurm log to appear and exec a pager on it. Pager command
	comes from $QUIET_STAR_FOLLOW_PAGER (default `less +F`); falls back to
	`tail -f` if the pager isn't on PATH (less isn't always installed). Not
	$PAGER — that's for static reading and rarely set up for live files."""
	pattern = slurm_params.get( "output", "slurm-%j.out" )
	log_path = (
		pattern
		.replace( "%j", str( job_id ) )
		.replace( "%x", str( slurm_params.get( "job_name", "qstar" ) ) ) )
	if not os.path.isabs( log_path ):
		log_path = os.path.join( dispatch_cwd, log_path )
	log_path = os.path.abspath( log_path )
	print( f"Waiting for {log_path}...", file = sys.stderr )
	while not os.path.exists( log_path ):
		time.sleep( 2 )

	pager_argv = shlex.split( os.environ.get( "QUIET_STAR_FOLLOW_PAGER", "less +F" ) )
	if shutil.which( pager_argv[ 0 ] ) is None:
		print(
			f"[follow] {pager_argv[ 0 ]} not on PATH, falling back to `tail -f`",
			file = sys.stderr )
		pager_argv = [ "tail", "-f" ]
	pager_argv.append( log_path )
	os.execvp( pager_argv[ 0 ], pager_argv )


_interrupted = False
_received_sigterm = False


def _graceful_exit_handler( signum, frame ):
	"""Set flag on first interrupt; ignore subsequent signals so checkpoint save
	and manifest write can complete. SIGKILL via scancel is still an escape hatch.
	Tracks SIGTERM separately so auto-resume only fires on Slurm preempt, not on
	SIGINT (Ctrl-C, wandb stop button, etc.)."""
	global _interrupted, _received_sigterm
	signal.signal( signal.SIGINT, signal.SIG_IGN )
	signal.signal( signal.SIGTERM, signal.SIG_IGN )
	_interrupted = True
	if signum == signal.SIGTERM:
		_received_sigterm = True
	name = signal.Signals( signum ).name
	print( f"\n{name} received. Will stop after current step...", file = sys.stderr )


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
	from transformers.integrations import WandbCallback
	from transformers.trainer_callback import ProgressCallback, PrinterCallback

	# Route tqdm progress through Python logging in non-tty mode so the Slurm
	# log gets nice periodic progress lines instead of CR-laden tty-style
	# updates. Monkey-patch must happen before HF's ProgressCallback is
	# constructed (it captures the module-level tqdm at instantiation time).
	try:
		from tqdm_loggable.auto import tqdm as _loggable_tqdm
		from tqdm_loggable.tqdm_logging import tqdm_logging as _tqdm_logging
		import logging
		import transformers.trainer_callback as _tc_mod
		_tc_mod.tqdm = _loggable_tqdm
		_tqdm_logging.set_log_rate( timedelta( seconds = 30 ) )
		# Give tqdm_loggable its own handler so its INFO-level output is visible
		# regardless of root logger configuration; propagate=False avoids
		# double-emitting if a root handler is also present.
		_tqdm_logger = logging.getLogger( "tqdm_loggable" )
		_tqdm_logger.setLevel( logging.INFO )
		if not _tqdm_logger.handlers:
			_h = logging.StreamHandler( sys.stderr )
			_h.setFormatter( logging.Formatter( "%(asctime)s %(message)s" ) )
			_tqdm_logger.addHandler( _h )
			_tqdm_logger.propagate = False
		_tqdm_loggable_ok = True
	except ImportError:
		_tqdm_loggable_ok = False
	from liger_kernel.transformers import AutoLigerKernelForCausalLM
	from datasets import load_dataset
	from quiet_star.eval_helpers import preprocess_function

	torch.backends.cuda.matmul.allow_tf32 = True

	# Opt-in memory profiler: set QUIET_STAR_MEMORY_PROFILE=1 to record per-step
	# allocator history and dump snapshot.pickle. Off by default — the recorder
	# adds CPU overhead and per-step pickle writes that are pure waste in
	# steady-state training.
	_memory_profile = os.environ.get( "QUIET_STAR_MEMORY_PROFILE" ) == "1"

	# Opt-in kernel-level profiler: set QUIET_STAR_TORCH_PROFILE=1 to wrap
	# training in torch.profiler. Records one full optimizer step (all grad-
	# accumulation micro-batches) after a 2-step wait + 1-step warmup, then
	# prints a top-30 kernel table sorted by CUDA time and writes a Chrome
	# trace to <output_dir>/torch_profile/trace.json. Rank 0 only. Does not
	# auto-stop training — scancel after the table prints to the slurm log.
	_torch_profile = os.environ.get( "QUIET_STAR_TORCH_PROFILE" ) == "1"

	@contextlib.contextmanager
	def cm_memory():
		if not _memory_profile:
			yield
			return
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
			# Durable sidecar so --resume <local-dir> can recover the wandb run id
			# even after the uploader job has cleared the pending_upload.json manifest.
			sidecar = {
				"run_id": wandb.run.id,
				"project": wandb.run.project,
				"entity": wandb.run.entity,
			}
			with open( os.path.join( args.output_dir, "wandb_run.json" ), "w" ) as f:
				json.dump( sidecar, f, indent = 2 )

	class GracefulStopCallback( TrainerCallback ):
		def __init__( self ):
			self._trainer = None
		def bind( self, trainer ):
			self._trainer = trainer
		def _save_and_exit( self, args, state, control ):
			# Save checkpoint, write a manifest for the next uploader run, and (only
			# on SIGTERM) write the auto-resume marker so the next requeue picks up
			# from this checkpoint. Skip the in-process wandb upload — it blows past
			# Slurm's ~5-min preemption grace period; the next training run picks up
			# the manifest at startup and submits an uploader sbatch concurrently.
			# Exit non-zero so the bash trap takes the "preserve worktree" branch
			# (rc=0 = clean training completion → trap removes worktree). 143 is the
			# conventional SIGTERM exit code (128 + 15).
			print( "Saving checkpoint and exiting...", file = sys.stderr )
			self._trainer._save_checkpoint( self._trainer.model, trial = None )
			if state.is_world_process_zero:
				self._write_manifest( args, state )
			# os._exit skips atexit (notably wandb's flush, which can block on uploads).
			os._exit( 143 )
		def _write_manifest( self, args, state ):
			import wandb
			run = wandb.run
			if run is None:
				print( "[graceful-stop] wandb.run is None; skipping manifest", file = sys.stderr )
				return
			ckpt_dir = os.path.abspath( os.path.join( args.output_dir, f"checkpoint-{state.global_step}" ) )
			# Mirror HF WandbCallback.on_save's artifact name/aliases/metadata so a
			# resumed run sees the same artifact as if we had uploaded in-process.
			artifact_name = (
				f"model-{run.id}"
				if ( args.run_name is None or args.run_name == args.output_dir )
				else f"model-{run.name}" )
			import numbers
			metadata = {
				k: v for k, v in dict( run.summary ).items()
				if isinstance( v, numbers.Number ) and not k.startswith( "_" ) }
			num_params = run.config.get( "model/num_parameters" )
			if num_params is not None:
				metadata[ "model/num_parameters" ] = num_params
			manifest = {
				"checkpoint_dir": ckpt_dir,
				"wandb": {
					"run_id": run.id,
					"run_name": run.name,
					"project": run.project,
					"entity": run.entity,
					"artifact_name": artifact_name,
					"artifact_type": "model",
					"metadata": metadata,
					"aliases": [
						f"epoch_{round( state.epoch, 2 )}",
						f"checkpoint_global_step_{state.global_step}",
					],
				},
			}
			with open( os.path.join( ckpt_dir, _PENDING_MANIFEST ), "w" ) as f:
				json.dump( manifest, f, indent = 2 )
			# Auto-resume marker is the SIGTERM-only signal that says "next run of
			# this same Slurm job should resume from here." SIGINT (Ctrl-C, wandb
			# stop) skips the marker so a manual cancel doesn't auto-resume.
			if _received_sigterm:
				with open( os.path.join( args.output_dir, _AUTO_RESUME_MARKER ), "w" ) as f:
					f.write( f"step {state.global_step}\n" )
			print(
				f"[graceful-stop] manifest written; auto_resume={_received_sigterm}; "
				f"output_dir={args.output_dir}",
				file = sys.stderr )
		# Check per substep: a full optimizer step can outlast Slurm's preemption grace period.
		def on_substep_end( self, args, state, control, **kwargs ):
			if _interrupted:
				self._save_and_exit( args, state, control )
			return control
		def on_step_end( self, args, state, control, **kwargs ):
			if _interrupted:
				self._save_and_exit( args, state, control )
			return control

	class SyncEmbeddingsCallback( TrainerCallback ):
		def on_step_end( self, args, state, control, model = None, **kwargs ):
			if model is not None:
				m = model.module if hasattr( model, "module" ) else model
				m.sync_thought_embeddings()
			return control

	class TorchProfilerCallback( TrainerCallback ):
		"""Opt-in torch.profiler wrapper gated on QUIET_STAR_TORCH_PROFILE=1.
		Schedule wait=2 / warmup=1 / active=1 / repeat=1: skips two optimizer
		steps for steady-state, takes one warmup step, records the next full
		optimizer step (with all grad-accum micro-batches), then transitions
		to NONE. on_trace_ready prints a top-30 kernel table sorted by CUDA
		time and writes a Chrome trace; rank 0 only. Does not auto-stop —
		scancel after the table prints (avoids the should_training_stop /
		DDP cross-rank hang)."""
		def __init__( self ):
			self._prof = None
			self._output_dir = None

		def on_train_begin( self, args, state, control, **kwargs ):
			if int( os.environ.get( "RANK", "0" ) ) != 0:
				return
			self._output_dir = args.output_dir
			self._prof = torch.profiler.profile(
				activities = [
					torch.profiler.ProfilerActivity.CPU,
					torch.profiler.ProfilerActivity.CUDA ],
				schedule = torch.profiler.schedule(
					wait = 2, warmup = 1, active = 1, repeat = 1 ),
				on_trace_ready = self._on_trace_ready,
				record_shapes = True,
			)
			self._prof.start()

		def on_step_end( self, args, state, control, **kwargs ):
			if self._prof is not None:
				self._prof.step()
			return control

		def on_train_end( self, args, state, control, **kwargs ):
			if self._prof is not None:
				self._prof.stop()
				self._prof = None

		def _on_trace_ready( self, prof ):
			out_dir = os.path.join( self._output_dir, "torch_profile" )
			os.makedirs( out_dir, exist_ok = True )
			trace_path = os.path.join( out_dir, "trace.json" )
			prof.export_chrome_trace( trace_path )
			table = prof.key_averages().table(
				sort_by = "cuda_time_total", row_limit = 30 )
			print(
				f"\n[torch-profile] top kernels by CUDA time:\n{table}\n"
				f"[torch-profile] chrome trace -> {trace_path}",
				file = sys.stderr )

	class DeepSpeedSafeWandbCallback( WandbCallback ):
		# Default WandbCallback.on_train_end builds a throwaway Trainer to call
		# save_model(); under DeepSpeed that constructs a second Accelerator with
		# the same deepspeed_plugin and Accelerate refuses. Reuse the live trainer.
		def __init__( self ):
			super().__init__()
			self._trainer = None

		def bind( self, trainer ):
			self._trainer = trainer

		def setup( self, args, state, model, **kwargs ):
			# When resuming, pre-init wandb with explicit id/resume args. Env vars
			# alone don't work: wandb caches its session settings on the first
			# wandb.Api() call (made by _resolve_resume), and any later changes to
			# WANDB_RUN_ID/RESUME are silently dropped — wandb prints a "session
			# already started" warning. Explicit init args bypass that caching.
			# Once wandb.run is non-None, the parent setup() skips init.
			if (
				state.is_world_process_zero
				and self._wandb.run is None
				and os.environ.get( "WANDB_RUN_ID" )
			):
				self._wandb.init(
					project = os.getenv( "WANDB_PROJECT", "huggingface" ),
					id = os.environ[ "WANDB_RUN_ID" ],
					resume = os.environ.get( "WANDB_RESUME", "must" ),
				)
			super().setup( args, state, model, **kwargs )

		def on_train_end( self, args, state, control, **kwargs ):
			if not ( self._initialized and state.is_world_process_zero ):
				return
			if not self._log_model.is_enabled or self._trainer is None:
				return
			import wandb
			name = "best" if args.load_best_model_at_end else "last"
			output_dir = os.path.join( args.output_dir, name )
			self._trainer.save_model( output_dir )
			artifact = wandb.Artifact( name = f"model-{wandb.run.id}", type = "model" )
			artifact.add_dir( output_dir )
			wandb.run.log_artifact( artifact, aliases = [ name ] )

	signal.signal( signal.SIGINT, _graceful_exit_handler )
	signal.signal( signal.SIGTERM, _graceful_exit_handler )

	torch.manual_seed( config["random_seed"] )
	random.seed( config["random_seed"] )

	os.environ[ "WANDB_PROJECT" ] = config["project_name"] + "-" + config["dataset"]["name"].split( "/" )[ -1 ]
	os.environ[ "WANDB_LOG_MODEL" ] = "checkpoint"

	tokenizer = AutoTokenizer.from_pretrained(
		config["base_model"]["name"],
		padding_side = config["base_model"]["tokenizer_padding_side"],
	)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	def model_init( p ):
		params = copy.deepcopy( config["thought_model"] )
		if p is not None:
			params |= p

		print( params )

		dtype = resolve_torch_dtype( config["base_model"]["torch_dtype"] )

		print( "Loading model" )

		# DeepSpeed manages device placement; device_map conflicts with it
		dm = None if training_args.deepspeed else config["base_model"]["device_map"]
		lm_model = AutoLigerKernelForCausalLM.from_pretrained(
			config["base_model"]["name"],
			torch_dtype = dtype,
			device_map = dm,
		)

		print( "Loaded model" )

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
		partial( preprocess_function, tokenizer = tokenizer, max_length = ds_cfg["train_snippet_length"] ),
		batched = True, writer_batch_size = 200 )
	print( "Loaded datasets" )

	tr = copy.deepcopy( config["training"] )

	# Pop custom keys that aren't TrainingArguments fields
	effective_batch_size = tr.pop( "effective_batch_size" )
	eval_lm_batch_size = tr.pop( "eval_lm_batch_size" )

	# Compute derived TrainingArguments fields
	ts = int( time.time() )
	timestamp = datetime.fromtimestamp( ts )
	# Stable output_dir per Slurm job — survives requeue so auto-resume can find
	# the prior run's checkpoints. Falls back to ts for non-Slurm runs.
	job_key = os.environ.get( "SLURM_JOB_ID" ) or str( ts )
	tr["output_dir"] = config["root_prefix"] + f"cache/quietstar/{job_key}"
	tr["optim"] = "adamw_torch_fused" if torch.cuda.is_available() or torch.backends.mps.is_available() else "adamw_torch"
	# torchrun sets WORLD_SIZE; default to 1 for local single-process runs.
	world_size = int( os.environ.get( "WORLD_SIZE", "1" ) )
	per_step_batch = tr["per_device_train_batch_size"] * world_size
	if effective_batch_size % per_step_batch != 0:
		raise ValueError(
			f"effective_batch_size={effective_batch_size} is not divisible by"
			f" per_device_train_batch_size={tr['per_device_train_batch_size']} * world_size={world_size}"
		)
	tr["gradient_accumulation_steps"] = effective_batch_size // per_step_batch
	tr["per_device_eval_batch_size"] = tr["per_device_train_batch_size"]
	tr["run_name"] = f"n{config['thought_model']['n_thoughts']}_d{config['thought_model']['thought_depth']}_la{config['thought_model']['look_ahead']}_{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}_{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}"

	# Auto-resume + auto-uploader, only meaningful on rank 0 (avoid duplicate
	# sbatch submissions from DDP workers). Auto-resume gates on the SIGTERM-only
	# .auto_resume marker; auto-uploader runs whenever pending manifests exist
	# (regardless of how the prior run stopped — pending data is pending data).
	rank0 = int( os.environ.get( "RANK", "0" ) ) == 0
	if rank0 and resume_from is None:
		marker_path = os.path.join( tr["output_dir"], _AUTO_RESUME_MARKER )
		if os.path.isfile( marker_path ) and _has_checkpoint_subdirs( tr["output_dir"] ):
			print(
				f"[auto-resume] {tr['output_dir']} has SIGTERM marker + checkpoints; resuming",
				file = sys.stderr )
			resume_from = tr["output_dir"]
			os.remove( marker_path )  # consume so we don't auto-resume again on next start
	if rank0 and os.path.isdir( tr["output_dir"] ):
		pending = any(
			os.path.isfile( os.path.join( tr["output_dir"], e, _PENDING_MANIFEST ) )
			for e in os.listdir( tr["output_dir"] ) )
		if pending:
			cfg_path = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".uploader_config.json" )
			if os.path.exists( cfg_path ) and os.environ.get( "SLURM_JOB_ID" ):
				print(
					f"[auto-uploader] pending manifests in {tr['output_dir']}; submitting uploader",
					file = sys.stderr )
				try:
					submit_uploader( tr["output_dir"] )
				except Exception as e:
					print(
						f"[auto-uploader] sbatch failed: {e!r}. Manifests left for manual upload-pending.",
						file = sys.stderr )
			else:
				print(
					f"[auto-uploader] pending manifests in {tr['output_dir']} but no uploader configured; "
					f"run `quiet_star_train.py upload-pending {tr['output_dir']}` manually",
					file = sys.stderr )

	training_args = TrainingArguments( **tr )

	graceful_cb = GracefulStopCallback()
	cb_list = [ ConfigArtifactCallback( config ), graceful_cb, SyncEmbeddingsCallback() ]
	if _torch_profile:
		cb_list.append( TorchProfilerCallback() )
	trainer = TrainerWithCache(
		args = training_args,
		train_dataset = train_dataset,
		eval_dataset = [ "commonsense_qa" ],
		model_init = model_init,
		processing_class = tokenizer,
		eval_lm_batch_size = eval_lm_batch_size,
		callbacks = cb_list,
	)
	graceful_cb.bind( trainer )

	# Replace the auto-registered WandbCallback in place: appending instead would
	# move it to the end of the list, so user callbacks like ConfigArtifactCallback
	# would run before wandb.run is initialized and silently no-op.
	wandb_cb = DeepSpeedSafeWandbCallback()
	wandb_cb.bind( trainer )
	for i, cb in enumerate( trainer.callback_handler.callbacks ):
		if type( cb ) is WandbCallback:
			trainer.callback_handler.callbacks[ i ] = wandb_cb
			break
	else:
		trainer.callback_handler.add_callback( wandb_cb )

	# PrinterCallback dumps per-step metric dicts; wandb has them, drop it.
	trainer.remove_callback( PrinterCallback )

	# Replace ProgressCallback with a quiet subclass that suppresses on_log
	# (same metric-dict spam as PrinterCallback). Combined with the
	# tqdm-loggable monkey patch above, the bar still emits periodic progress
	# lines via Python logging — just without the per-step stats noise. If
	# tqdm-loggable isn't installed, drop the progress bar entirely (regular
	# tqdm in a non-tty Slurm log produces a wall of CR-laden garbage).
	class QuietProgressCallback( ProgressCallback ):
		def on_log( self, args, state, control, logs = None, **kwargs ):
			return
	if _tqdm_loggable_ok:
		for i, cb in enumerate( trainer.callback_handler.callbacks ):
			if type( cb ) is ProgressCallback:
				trainer.callback_handler.callbacks[ i ] = QuietProgressCallback()
				break
	else:
		print(
			"[setup] tqdm-loggable not installed; dropping progress bar. "
			"`pip install tqdm-loggable` for log-friendly progress.",
			file = sys.stderr )
		trainer.remove_callback( ProgressCallback )

	resume_checkpoint = None
	if resume_from is not None:
		print( f"[resume] WANDB_CACHE_DIR = {os.environ.get( 'WANDB_CACHE_DIR', '<unset>' )}", file = sys.stderr )
		if os.path.isdir( resume_from ):
			ckpt_dir, _, run_info = _resolve_local_resume( resume_from )
			os.environ[ "WANDB_RUN_ID" ] = run_info[ "run_id" ]
			os.environ[ "WANDB_RESUME" ] = "must"
			if run_info.get( "project" ):
				os.environ.setdefault( "WANDB_PROJECT", run_info[ "project" ] )
			if run_info.get( "entity" ):
				os.environ.setdefault( "WANDB_ENTITY", run_info[ "entity" ] )
			resume_checkpoint = ckpt_dir
			print( f"[resume] resuming from local path {ckpt_dir} (run_id={run_info['run_id']})", file = sys.stderr )
		else:
			artifact, run = _resolve_resume( resume_from )
			print( f"[resume] downloading artifact {artifact.name} (created {artifact.created_at})", file = sys.stderr )
			resume_checkpoint = artifact.download()
			print( f"[resume] artifact materialised at {resume_checkpoint}", file = sys.stderr )

	print("distributed_type:", trainer.accelerator.distributed_type)
	print("deepspeed_plugin:", trainer.accelerator.state.deepspeed_plugin)
	print( "[pre-train] WANDB_*:", { k: v for k, v in os.environ.items() if k.startswith( "WANDB_" ) }, file = sys.stderr )

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
	train_parser.add_argument( "--resume", metavar = "ARTIFACT_OR_PATH", default = None,
		help = "W&B checkpoint artifact, run path, or local checkpoint dir to resume from" )
	train_parser.add_argument( "--slurm", metavar = "SLURM_CONFIG", default = None,
		help = "Submit to Slurm using this config file instead of running locally" )
	train_parser.add_argument( "--root-prefix", metavar = "PATH", default = None,
		help = "Root path for cache/output dirs; overrides WANDB_CACHE_DIR and config root_prefix" )
	train_parser.add_argument( "--follow", "-f", action = "store_true",
		help = "After --slurm submit, exec `less +F` on the job's log file." )

	upload_parser = subparsers.add_parser( "upload-pending",
		help = "Upload checkpoints flagged by graceful-stop manifests" )
	upload_parser.add_argument( "output_dir", help = "Training output_dir to scan for pending manifests" )

	submit_uploader_parser = subparsers.add_parser( "submit-uploader",
		help = "(Internal) Submit an uploader sbatch job for the given pending dir" )
	submit_uploader_parser.add_argument( "pending_dir", help = "Output_dir containing pending_upload.json" )

	subparsers.add_parser( "run-uploader-job",
		help = "(Internal) Uploader sbatch job entrypoint; reads QUIET_STAR_PENDING_DIR" )

	args = parser.parse_args()

	if args.command == "init":
		save_config( DEFAULT_CONFIG, args.path )
		print( f"Wrote default config to {args.path}" )
	elif args.command == "init-slurm":
		with open( args.path, "w" ) as f:
			f.write( SLURM_TEMPLATE )
		print( f"Wrote Slurm template to {args.path}" )
	elif args.command == "upload-pending":
		upload_pending( args.output_dir )
	elif args.command == "submit-uploader":
		submit_uploader( args.pending_dir )
	elif args.command == "run-uploader-job":
		run_uploader_job()
	elif args.command == "train":
		if args.slurm is not None:
			submit_slurm( args.slurm, args )
		else:
			# WANDB_CACHE_DIR must be set before any wandb API call (including
			# config_from_wandb_checkpoint) so the cache used during the original
			# run matches the cache used on resume.
			path_cfg = load_config( args.path ) if args.path is not None else None
			cache_dir = _resolve_wandb_cache_dir( args.root_prefix, path_cfg )
			if cache_dir is not None:
				os.environ[ "WANDB_CACHE_DIR" ] = cache_dir
			print( f"[main] WANDB_CACHE_DIR = {os.environ.get( 'WANDB_CACHE_DIR', '<unset>' )}", file = sys.stderr )

			# When resuming, always use the checkpoint's config — --path is
			# consulted only for root_prefix above and then discarded.
			if args.resume is not None:
				if os.path.isdir( args.resume ):
					cfg = config_from_local_checkpoint( args.resume )
				else:
					cfg = config_from_wandb_checkpoint( args.resume )
			elif path_cfg is not None:
				cfg = path_cfg
			else:
				cfg = copy.deepcopy( DEFAULT_CONFIG )
			train( cfg, resume_from = args.resume )
