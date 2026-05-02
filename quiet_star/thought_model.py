from typing import override
import einx as x
import torch as t
import torch.utils.checkpoint as ckpt_utils
from transformers import GenerationMixin, PreTrainedModel, DynamicCache, StaticCache, PretrainedConfig, AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


class WeightedMixerHead( t.nn.Module ):
	def __init__( self, config, lm_hidden_size ):
		super().__init__()
		self.config = config
		hs, ms, nl = lm_hidden_size, config.mixer_config.hidden_size, config.mixer_config.hidden_layers
		layers = [ t.nn.Linear( 2 * hs, ms ) ]
		for _ in range( nl ):
			layers.extend( [ ACT2FN[ config.mixer_config.activation ], t.nn.Linear( ms, ms ) ] )
		layers.extend( [ ACT2FN[ config.mixer_config.activation ], t.nn.Linear( ms, 1 ) ] )

		self.mlp = t.nn.Sequential( *layers )

	def forward( self, pre_thought_hidden_state, post_thought_hidden_state ):
		catted_states = t.cat( (pre_thought_hidden_state, post_thought_hidden_state), dim = -1 )
		return t.sigmoid(self.mlp( catted_states ))


# class ConfidenceHead( t.nn.Module ):
# 	def __init__( self, config ):
# 		super().__init__()
# 		hs, ms, nl = config.hidden_size, config.confidence_head_hidden_size, config.confidence_head_hidden_layers
# 		layers = [ t.nn.Linear( 2 * hs + 1, ms ) ]
# 		for _ in range( nl ):
# 			layers.extend( [ t.nn.Linear( ms, ms ), ACT2FN[ config.confidence_head_activation ] ] )
# 		layers.extend(
# 			[ ACT2FN[ config.confidence_head_activation ], t.nn.Linear( ms, 1 ),
# 				ACT2FN[ config.confidence_head_output_activation ] ] )
#
# 		self.mlp = t.nn.Sequential( *layers )
#
# 	def forward( self, pre_thought_hidden_state, post_thought_hidden_state, mixer_value ):
# 		catted_states = t.cat( (pre_thought_hidden_state, post_thought_hidden_state, mixer_value), dim = -1 )
# 		return self.mlp( catted_states )

class MixerConfig:
	def __init__(
			self,
			*,
			activation: str = "relu",
			hidden_layers: int = 1,
			hidden_size: int = None,
			mixer_type: str = "weighted",
			_inject_hidden_size: int = None,
	):
		if mixer_type != "weighted":
			raise ValueError( f"`mixer_type` must be `weighted`, got {mixer_type}" )

		self.activation = activation
		self.hidden_layers = hidden_layers
		self.mixer_type = mixer_type

		self._hidden_size = hidden_size
		self._injected_hidden_size = _inject_hidden_size

	@property
	def hidden_size( self ):
		if self._hidden_size is not None:
			return self._hidden_size

		return self._injected_hidden_size

	@hidden_size.setter
	def hidden_size( self, value ):
		self._hidden_size = value

	def to_dict( self ):
		out = {
			"mixer_type": self.mixer_type,
			"activation": self.activation,
			"hidden_layers": self.hidden_layers,
		}

		if self._hidden_size is not None:
			out[ "hidden_size" ] = self._hidden_size

		return out

	def items( self ):
		yield from self.to_dict().items()

	def __iter__( self ):
		yield from self.to_dict().items()


def nanmin( x ):
	return x[ ~x.isnan() ].min()


def nanmax( x ):
	return x[ ~x.isnan() ].max()


def nancorr( a, b ):
	"""Pearson correlation across all elements, ignoring positions where either is NaN."""
	a = a.flatten().float()
	b = b.flatten().float()
	mask = ~(a.isnan() | b.isnan())
	a, b = a[ mask ], b[ mask ]
	if a.numel() < 2:
		return t.tensor( float( "nan" ), device = a.device )
	a_c = a - a.mean()
	b_c = b - b.mean()
	denom = a_c.norm() * b_c.norm()
	if denom == 0:
		return t.tensor( float( "nan" ), device = a.device )
	return (a_c * b_c).sum() / denom


class ThoughtModelConfig( PretrainedConfig ):
	model_type = "thought_model"
	is_composition = True
	has_no_defaults_at_init = True
	sub_configs = { "mixer_config": MixerConfig, "text_config": AutoConfig }

	def __init__(
			self,
			*,
			beta_mixed: float = 1.0,
			beta_stability: float = 1.0,
			beta_thought: float = 1.0,
			end_thought_token_id: int = None,
			embedding_scale: float = 1.0,
			coef_entropy: float = 0.0,
			gated_reinforce: bool = False,
			mixer_init_bias: float = -5.0,
			mixer_zero_init: bool = False,
			stt_init_id: int = None,
			ett_init_id: int = None,
			look_ahead: int = 4,
			look_ahead_pass: bool = False,
			n_thoughts: int = 2,
			pad_token_id: int = None,
			reinforce_temperature: float = 3.0,
			start_thought_token_id: int = None,
			thought_depth: int = 12,
			thought_temperature: float = 1.0,

			mixer_config: dict | MixerConfig = None,
			text_config: dict | PretrainedConfig = None,
			**kwargs ):

		self.beta_mixed = beta_mixed
		self.beta_stability = beta_stability
		self.beta_thought = beta_thought
		self.end_thought_token_id = end_thought_token_id
		self.embedding_scale = embedding_scale
		self.coef_entropy = coef_entropy
		self.gated_reinforce = gated_reinforce
		self.mixer_init_bias = mixer_init_bias
		self.mixer_zero_init = mixer_zero_init
		self.stt_init_id = stt_init_id
		self.ett_init_id = ett_init_id
		self.look_ahead = look_ahead
		self.look_ahead_pass = look_ahead_pass
		self.n_thoughts = n_thoughts
		self.pad_token_id = pad_token_id
		self.reinforce_temperature = reinforce_temperature
		self.start_thought_token_id = start_thought_token_id
		self.thought_depth = thought_depth
		self.thought_temperature = thought_temperature

		if text_config is None:
			raise ValueError( "`text_config` must be specified" )
		elif isinstance( text_config, dict ):
			text_config[ "hidden_size" ] = kwargs.get( "hidden_size", None )
			if "model_type" not in text_config:
				raise ValueError( f"`lm_config` must have `model_type`" )
			self.text_config = AutoConfig.for_model( text_config[ "model_type" ], **text_config )
		else:
			self.text_config = text_config

		if mixer_config is None:
			self.mixer_config = MixerConfig( _inject_hidden_size = self.text_config.hidden_size )
		elif isinstance( mixer_config, dict ):
			self.mixer_config = MixerConfig( **mixer_config, _inject_hidden_size = self.text_config.hidden_size )
		else:
			self.mixer_config = mixer_config

		super().__init__( **kwargs )


AutoConfig.register( "thought_model", ThoughtModelConfig )


class ThoughtModel( PreTrainedModel, GenerationMixin ):
	config_class = ThoughtModelConfig
	_supports_cache_class = True
	_supports_static_cache = True
	supports_gradient_checkpointing = True
	base_model_prefix = "lm_model"

	# def _init_weights( self, module ):
	# 	std = self.config.initializer_range
	# 	if isinstance( module, t.nn.Linear ):
	# 		module.weight.data.normal_( mean = 0.0, std = std )
	# 		if module.bias is not None:
	# 			module.bias.data.zero_()

	def __init__( self, config: ThoughtModelConfig, *, lm_model = None ):
		super().__init__( config )

		self.start_thought_token_id = config.start_thought_token_id
		self.end_thought_token_id = config.end_thought_token_id
		self.pad_token_id = config.pad_token_id

		self.coef_entropy = config.coef_entropy
		self.beta_mixed = config.beta_mixed
		self.beta_stability = config.beta_stability
		self.beta_thought = config.beta_thought
		self.gated_reinforce = config.gated_reinforce
		self.look_ahead = config.look_ahead
		self.look_ahead_pass = config.look_ahead_pass
		self.mixer_config = config.mixer_config
		self.n_thoughts = config.n_thoughts
		self.reinforce_temperature = config.reinforce_temperature
		self.thought_depth = config.thought_depth
		self.thought_temperature = config.thought_temperature

		self.lm_model = lm_model if lm_model is not None else AutoModel.from_config( config.text_config )
		self.mixer_head = WeightedMixerHead( config, lm_model.config.hidden_size ).to( dtype = self.lm_model.dtype )

		self._flce = LigerFusedLinearCrossEntropyLoss(
			ignore_index = self.pad_token_id if self.pad_token_id is not None else -100,
			reduction = "none",
		)

		self.embedding_scale = config.embedding_scale
		hidden_size = self.lm_model.config.hidden_size
		self.start_embedding = t.nn.Parameter( t.zeros( hidden_size, dtype = self.lm_model.dtype ) )
		self.end_embedding = t.nn.Parameter( t.zeros( hidden_size, dtype = self.lm_model.dtype ) )

		self.post_init()

	@override
	def post_init(self):
		super().post_init()

		with t.no_grad():
			last_layer = self.mixer_head.mlp[ -1 ]
			if self.config.mixer_zero_init:
				last_layer.weight.zero_()
			last_layer.bias.fill_( self.config.mixer_init_bias )

			if self.config.stt_init_id is not None:
				self.start_embedding.copy_(
					self.lm_model.model.embed_tokens.weight[ self.config.stt_init_id ] / self.embedding_scale )
			if self.config.ett_init_id is not None:
				self.end_embedding.copy_(
					self.lm_model.model.embed_tokens.weight[ self.config.ett_init_id ] / self.embedding_scale )

			self.sync_thought_embeddings()


	def sync_thought_embeddings( self ):
		"""Write scaled thought embeddings into the embedding table."""
		with t.no_grad():
			self.lm_model.model.embed_tokens.weight[ self.start_thought_token_id ] = self.start_embedding * self.embedding_scale
			self.lm_model.model.embed_tokens.weight[ self.end_thought_token_id ] = self.end_embedding * self.embedding_scale

	def _embed( self, input_ids ):
		"""Look up embeddings, differentiably replacing thought token positions."""
		embeds = self.lm_model.model.embed_tokens( input_ids )
		stt_mask = (input_ids == self.start_thought_token_id).unsqueeze( -1 )
		ett_mask = (input_ids == self.end_thought_token_id).unsqueeze( -1 )
		embeds = t.where( stt_mask, self.start_embedding * self.embedding_scale, embeds )
		embeds = t.where( ett_mask, self.end_embedding * self.embedding_scale, embeds )
		return embeds

	@staticmethod
	def construct_thought_mask( b, n, d, l, padding_mask, dtype ):
		device = padding_mask.device

		padding_mask = padding_mask.bool()

		range_d = x.rearrange( "d -> D L d l", t.arange( d, device = device ), l = l, d = d, D = d, L = l )
		range_D = x.rearrange( "D -> D L d l", t.arange( d, device = device ), l = l, d = d, D = d, L = l )
		range_l = x.rearrange( "l -> D L d l", t.arange( l, device = device ), l = l, d = d, D = d, L = l )
		range_L = x.rearrange( "L -> D L d l", t.arange( l, device = device ), l = l, d = d, D = d, L = l )

		# Allow attention only for tokens which precede in depth, or are in the initial input sequence and precede in time
		countercausal_mask = ((range_d <= range_D) & (range_l == range_L)) | ((range_d == 0) & (range_l <= range_L))
		countercausal_mask = x.rearrange(
			"D L d l -> b n D L d l", countercausal_mask, b = b, n = n, d = d, l = l, D = d, L = l )
		padding_mask = x.rearrange( "b n l -> b n D L d l", padding_mask, b = b, n = n, d = d, l = l, D = d, L = l )

		combined_mask = countercausal_mask & padding_mask

		# Invert the mask and return
		return t.zeros_like( combined_mask, dtype = dtype, device = device ).masked_fill(
			~combined_mask, t.finfo( dtype ).min )

	@staticmethod
	def sample_thoughts( logits, thought_temperature ):
		if thought_temperature != 0.0:
			logits = t.nn.functional.gumbel_softmax( logits, tau = thought_temperature, hard = True, dim = -1 )

		return t.argmax( logits, dim = -1 )

	def prepare_cache_positions( self, d_bot, d_top, l, device = None ):
		if device is None:
			device = self.device
		layers = t.arange( d_bot, d_top, device = device )
		return t.arange( l, device = device ) + l * layers.unsqueeze( -1 )

	def _flce_loss( self, hidden, targets, *, scale = 1.0 ):
		"""Fused linear + CE on hidden states. Returns per-token loss shaped like targets."""
		e = hidden.shape[ -1 ]
		h = hidden.reshape( -1, e )
		if scale != 1.0:
			h = h / scale
		bias = getattr( self.lm_model.lm_head, "bias", None )
		return self._flce(
			self.lm_model.lm_head.weight, h, targets.reshape( -1 ), bias
		).reshape_as( targets )

	def broadcast_forward( self, ts, kv_cache, layers_cached, layer_to_gen, mask, keep = 1, *, compute_logits = True ):
		b, n, d, l = ts.shape

		if keep is None:
			keep = d

		if kv_cache is not None:
			cache_pos = x.rearrange( "d l -> (d l)", self.prepare_cache_positions( layers_cached, layer_to_gen, l ) )
		else:
			cache_pos = None

		a_mask = x.rearrange( "b n D L d l -> (b n) 1 (D L) (d l)", mask )
		input_ids = x.rearrange( "b n d l -> (b n) (d l)", ts )
		inputs_embeds = self._embed( input_ids )
		position_ids = x.rearrange(
			"d l -> (b n) (d l)", t.arange( l, device = ts.device ) + (t.arange( d, device = ts.device ) + layers_cached).unsqueeze( -1 ),
			b = b, n = n )

		if compute_logits:
			out = self.lm_model(
				inputs_embeds = inputs_embeds,
				position_ids = position_ids,
				attention_mask = a_mask,
				past_key_values = kv_cache,
				use_cache = kv_cache is not None,
				output_attentions = False,
				output_hidden_states = True,
				return_dict = True,
				cache_position = cache_pos,
				logits_to_keep = l * keep,
			)

			log = x.rearrange( "(b n) (d l) v -> b n d l v", out.logits, b = b, n = n, d = keep, l = l )
			hidden = x.rearrange( "(b n) (d l) e -> b n d l e", out.hidden_states[ -1 ], b = b, n = n, d = d, l = l )[ ...,
			-keep:, :, : ]

			return log, hidden
		else:
			out = self.lm_model.model(
				inputs_embeds = inputs_embeds,
				position_ids = position_ids,
				attention_mask = a_mask,
				past_key_values = kv_cache,
				use_cache = kv_cache is not None,
				output_attentions = False,
				output_hidden_states = False,
				return_dict = True,
				cache_position = cache_pos,
			)

			hidden = x.rearrange( "(b n) (d l) e -> b n d l e", out.last_hidden_state, b = b, n = n, d = d, l = l )[ ...,
			-keep:, :, : ]

			return None, hidden

	def naive_forward( self, input_ids, padding_mask, kv_cache = None, cache_pos = None, keep = 1, *, compute_logits = True ):
		assert cache_pos is None or input_ids.shape[ -1 ] == cache_pos.shape[ -1 ]
		# Removed the padding_mask length sanity check that referenced
		# kv_cache.get_seq_length(): under StaticCache get_seq_length() counts
		# non-zero positions, which becomes nonzero after the first forward
		# and trips the assert during gradient-checkpoint recomputation in
		# backward (the cache state has changed since the original forward).
		# The check was only meaningful under DynamicCache's append semantics;
		# with explicit cache_position past-seen-tokens comes from cache_pos[0]
		# and Mistral handles the mask sizing internally.

		if self.training:
			model_input = { "inputs_embeds": self._embed( input_ids ) }
		else:
			model_input = { "input_ids": input_ids }

		if compute_logits:
			out = self.lm_model(
				**model_input,
				attention_mask = padding_mask,
				past_key_values = kv_cache,
				use_cache = kv_cache is not None,
				output_attentions = False,
				output_hidden_states = True,
				return_dict = True,
				cache_position = cache_pos,
				logits_to_keep = keep,
			)

			log = out.logits
			hidden = out.hidden_states[ -1 ][ ..., -keep:, : ]

			return log, hidden
		else:
			out = self.lm_model.model(
				**model_input,
				attention_mask = padding_mask,
				past_key_values = kv_cache,
				use_cache = kv_cache is not None,
				output_attentions = False,
				output_hidden_states = False,
				return_dict = True,
				cache_position = cache_pos,
			)

			hidden = out.last_hidden_state[ ..., -keep:, : ]

			return None, hidden

	def _maybe_ckpt( self, fn, *args, **kwargs ):
		"""Conditionally wrap fn in torch.utils.checkpoint.checkpoint, gated on
		self._grad_ckpt (set externally via QUIET_STAR_GRAD_CKPT). Drops
		intermediate activations from fn's forward graph and recomputes them
		during backward — trades compute for memory.

		Safe for our pattern because:
		  - StaticCache writes are at fixed slots via cache_position, idempotent
		    under re-execution. Re-running a checkpointed broadcast_forward
		    rewrites the same K/V to the same slots.
		  - The thought-loop attention mask restricts each call's reads to
		    depths causal relative to that call. Later calls' writes to the
		    same cache buffer don't pollute the recomputation (those positions
		    are masked out at -inf).

		use_reentrant=False is required — the reentrant variant interacts
		badly with autograd hooks under DDP / DeepSpeed."""
		if getattr( self, "_grad_ckpt", False ):
			return ckpt_utils.checkpoint( fn, *args, use_reentrant = False, **kwargs )
		return fn( *args, **kwargs )

	def _ensure_static_caches( self, b, L, device ):
		"""Lazy-allocate the two StaticCaches on first training_forward call.
		Subsequent calls reuse the buffers — every position is overwritten
		by the prior pass + K/V copy + depth loop + look-ahead, so previous-
		call state doesn't leak into the current call's attention. Cache
		allocation has to live outside any torch.compile region because
		StaticCache.__init__ calls torch._dynamo.mark_static_address, which
		Dynamo refuses to trace."""
		if getattr( self, "_prior_cache", None ) is not None:
			return
		n = self.n_thoughts
		l = L - self.look_ahead
		d = 2 + self.thought_depth + self.look_ahead
		text_config = self.lm_model.config
		dtype = self.lm_model.dtype
		self._prior_cache = StaticCache(
			config = text_config,
			max_batch_size = b,
			max_cache_len = L - 1,
			dtype = dtype,
			device = device,
		)
		self._broadcast_cache = StaticCache(
			config = text_config,
			max_batch_size = b * n,
			max_cache_len = d * l,
			dtype = dtype,
			device = device,
		)

	def training_forward( self, input_ids, labels, padding_mask, thought_temperature, *, kv_cache = None ):
		"""Dispatcher for end-to-end torch.compile of the training forward.

		Allocates the two StaticCaches on first call (outside the compile
		region — Dynamo can't trace StaticCache.__init__'s mark_static_address
		call), then routes to _training_forward_impl. Under
		QUIET_STAR_COMPILE_FULL, the first call runs the eager impl to warm
		up einx pattern caches (einx generates and caches torch ops the
		first time a pattern is seen with a given shape; Dynamo can't trace
		through the cache-miss path), and subsequent calls use the compiled
		wrapper. When _compile_mode_full is None, this is just a thin
		wrapper around _training_forward_impl that handles cache allocation."""
		self._ensure_static_caches( input_ids.shape[ 0 ], input_ids.shape[ 1 ], input_ids.device )
		impl_kwargs = dict(
			kv_cache = kv_cache,
			prior_cache = self._prior_cache,
			broadcast_cache = self._broadcast_cache,
		)
		if getattr( self, "_compile_mode_full", None ) is None:
			return self._training_forward_impl( input_ids, labels, padding_mask, thought_temperature, **impl_kwargs )
		if not getattr( self, "_einx_warmed", False ):
			self._einx_warmed = True
			return self._training_forward_impl( input_ids, labels, padding_mask, thought_temperature, **impl_kwargs )
		if getattr( self, "_compiled_training_forward", None ) is None:
			self._compiled_training_forward = t.compile(
				self._training_forward_impl,
				mode = self._compile_mode_full,
			)
		return self._compiled_training_forward( input_ids, labels, padding_mask, thought_temperature, **impl_kwargs )

	def _training_forward_impl(
			self, input_ids, labels, padding_mask, thought_temperature,
			*, kv_cache = None, prior_cache = None, broadcast_cache = None ):
		# kv_cache argument is retained for API compatibility but unused. The
		# two StaticCaches are allocated outside this function (in the
		# training_forward dispatcher) and passed in — Dynamo refuses to
		# trace StaticCache.__init__'s mark_static_address call. The thought-
		# loop attention mask is sized for d*l KV positions; with StaticCache
		# the cache tensor returned to attention is also d*l (zero-filled at
		# unwritten slots, masked out by -inf in the additive mask), so
		# shapes match exactly without any per-call reshape or append.
		assert input_ids.device == padding_mask.device
		b, L = input_ids.shape
		n = self.n_thoughts
		l = L - self.look_ahead
		d = 2 + self.thought_depth + self.look_ahead
		text_config = self.lm_model.config
		dtype = self.lm_model.dtype
		device = input_ids.device

		# Prior LM forward — populates prior_cache at positions [0, L-1).
		cache_pos_prior = t.arange( L - 1, device = device )
		_, prior_hidden_states = self._maybe_ckpt(
			self.naive_forward,
			input_ids[ :, :-1 ], padding_mask[ :, :-1 ],
			kv_cache = prior_cache, cache_pos = cache_pos_prior,
			keep = input_ids.shape[ -1 ], compute_logits = False )
		# Clone before later compiled calls overwrite the CUDA Graph static
		# output buffer this view points into. Required under mode=reduce-overhead;
		# wasteful (~10 us / step) but harmless under mode=default.
		prior_hidden_states = prior_hidden_states.clone()
		prior_hidden_states = prior_hidden_states.unfold( -2, self.look_ahead, 1 )
		prior_hidden_states = x.rearrange( "b l e d -> b n d l e", prior_hidden_states, n = n )

		# Copy depth-0 K/V from prior_cache to broadcast_cache, replicating
		# batch dim from b to b*n. Only the first l = L-look_ahead positions
		# are needed in broadcast_cache (the broadcast layout's depth-0 slot).
		for i in range( text_config.num_hidden_layers ):
			k = prior_cache.key_cache[ i ][ :, :, :l, : ]
			v = prior_cache.value_cache[ i ][ :, :, :l, : ]
			broadcast_cache.key_cache[ i ][ :, :, :l, : ] = k.repeat_interleave( n, dim = 0 )
			broadcast_cache.value_cache[ i ][ :, :, :l, : ] = v.repeat_interleave( n, dim = 0 )

		# Truncate the input by look_ahead
		truncated_input = input_ids[ ..., :-self.look_ahead ]

		# Preallocate ts at the full d-layer layout. Slot layout:
		#   [0]                      input prefix (b broadcast across n)
		#   [1]                      start_thought_token_id
		#   [2 .. 1+thought_depth]   sampled thought tokens (filled in the loop)
		#   [2+thought_depth]        end_thought_token_id
		#   [3+thought_depth .. d-1] look_ahead-1 target tokens (teacher forcing)
		# Replaces the original cat-grow pattern (4 separate t.cat allocations).
		ts = t.empty( (b, n, d, l), dtype = truncated_input.dtype, device = device )
		ts[ :, :, 0, : ] = truncated_input.unsqueeze( 1 )
		ts[ :, :, 1, : ] = self.start_thought_token_id

		# rank-6 mask of shape (b, n, d, l, d, l)
		thought_mask = self.construct_thought_mask(
			b = b,
			n = n,
			d = d,  # input (1), start tok (1), depth, end tok (1), look ahead (-1 for teacher forcing shift)
			l = l,
			padding_mask = x.rearrange( "b l -> b n l", padding_mask, n = n )[ ..., :-self.look_ahead ],
			dtype = self.dtype )

		# Preallocate per-step accumulators. Replaces the in-loop t.cat that was
		# O(thought_depth^2) memcpy and would force per-iter Dynamo recompiles.
		e = self.lm_model.config.hidden_size
		thought_hidden_states = t.empty( (b, n, self.thought_depth, l, e), dtype = dtype, device = device )
		thought_entropy_acc = t.empty( (b, n, self.thought_depth, l), dtype = dtype, device = device )

		# Generate thoughts (depth 0 already cached from naive_forward)
		# We accumulate hidden states (for FLCE later) and per-slice entropy (for logging).
		# lm_head is applied under no_grad for sampling/entropy only — the thought REINFORCE
		# gradient flows through thought_hidden_states via FLCE later, so keeping lm_head out
		# of the autograd graph here saves ~b*n*l*vocab activation per depth step.
		layers_cached, layer_to_gen = 1, 2
		for i in range( self.thought_depth ):
			slice_mask = thought_mask[ :, :, layers_cached: layer_to_gen, :, :, : ]

			# .clone() on the input slot — embed_tokens.backward saves the input_ids
			# tensor for its weight-gradient scatter, and we mutate ts in-place
			# inside this loop (sampled tokens written to subsequent slots). Without
			# the clone, ts's version bumps invalidate the saved views from earlier
			# iterations.
			_, new_hidden = self._maybe_ckpt(
				self.broadcast_forward,
				ts[ :, :, layers_cached: layer_to_gen, : ].clone(),
				broadcast_cache,
				layers_cached,
				layer_to_gen,
				slice_mask,
				compute_logits = False )

			thought_hidden_states[ :, :, i: i + 1, :, : ] = new_hidden

			with t.no_grad():
				new_logs = self.lm_model.lm_head( new_hidden )
				new_toks = self.sample_thoughts( new_logs, thought_temperature )
				slice_log_probs = t.nn.functional.log_softmax( new_logs / self.reinforce_temperature, dim = -1 )
				slice_entropy = -(slice_log_probs.exp() * slice_log_probs).sum( dim = -1 )
				thought_entropy_acc[ :, :, i: i + 1, : ] = slice_entropy

			del new_logs

			layers_cached = layer_to_gen
			layer_to_gen += 1

			# Sampled token goes into the slot it just generated.
			ts[ :, :, layer_to_gen - 1: layer_to_gen, : ] = new_toks

		# Add </thought> at slot 2 + thought_depth.
		ts[ :, :, 2 + self.thought_depth, : ] = self.end_thought_token_id
		layer_to_gen += 1

		# Prepare the sliding targets
		targets = labels.unfold( -1, self.look_ahead, 1 )[ :, 1: ].transpose( -1, -2 )
		targets = x.rearrange( "b d l -> b n d l", targets, n = n, d = self.look_ahead )

		# Write targets[:, :, :-1, :] into the trailing slots (teacher forcing shift
		# drops the last target — its hidden state isn't used).
		ts[ :, :, 3 + self.thought_depth: d, : ] = targets[ :, :, :-1, : ]

		# Hidden states with thought (no lm_head — losses are fused via FLCE below)
		if self.look_ahead_pass:
			# Sequential look-ahead: process one look-ahead position per LM forward pass.
			# Equivalent to the batched path (KV cache makes it mathematically identical),
			# but with smaller per-call activation memory.
			# At entry: layers_cached covers [0, last_thought), layer_to_gen one past end_tok.
			# - iter 0: slice = [last_thought, end_tok] (2 layers), keep=1 -> end_tok logit = label[l+1]
			# - iters 1..look_ahead-1: slice = [target_{i-1}] (1 layer), keep=1 -> label[l+i+1]
			post_hidden_states = None
			for i in range( self.look_ahead ):
				if i > 0:
					layer_to_gen += 1
				slice_mask = thought_mask[ :, :, layers_cached: layer_to_gen, :, :, : ]
				_, hidden = self._maybe_ckpt(
					self.broadcast_forward,
					ts[ :, :, layers_cached: layer_to_gen, : ],
					broadcast_cache,
					layers_cached,
					layer_to_gen,
					slice_mask,
					keep = 1,
					compute_logits = False )
				if i == 0:
					post_hidden_states = hidden
				else:
					post_hidden_states = t.cat( (post_hidden_states, hidden), dim = -3 )
				layers_cached = layer_to_gen
		else:
			layer_to_gen += self.look_ahead - 1  # -1 because targets have one fewer layer for teacher forcing shift
			slice_mask = thought_mask[ :, :, layers_cached: layer_to_gen, :, :, : ]
			_, post_hidden_states = self._maybe_ckpt(
				self.broadcast_forward,
				ts[ :, :, layers_cached: layer_to_gen, : ],
				broadcast_cache,
				layers_cached,
				layer_to_gen,
				slice_mask,
				keep = self.look_ahead,
				compute_logits = False )
			# We won't use this anymore, but update it as a matter of hygiene
			layers_cached = layer_to_gen
			layer_to_gen += 1

		# Mix at hidden level. lm_head is linear and alpha has shape (..., 1) (per-token scalar
		# gate broadcasting over hidden), so this is mathematically equivalent to mixing logits.
		prior_hidden_for_mix = prior_hidden_states.expand_as( post_hidden_states )
		alpha = self.mixer_head( prior_hidden_for_mix, post_hidden_states )
		mixed_hidden = alpha * post_hidden_states + (1 - alpha) * prior_hidden_for_mix

		mask_for_loss = x.rearrange( "b l -> b 1 1 l", (~padding_mask.bool()) )[ ..., :-self.look_ahead ]

		# Mixed CE
		mixed_cross_entropy_loss = self._flce_loss( mixed_hidden, targets )
		mixed_cross_entropy_loss = mixed_cross_entropy_loss.masked_fill( mask_for_loss, t.nan )
		mixed_cross_entropy_loss = x.reduce( "b n [d] l", mixed_cross_entropy_loss, op = t.nanmean )

		# Prior CE — same across thoughts, so the FLCE matmul is computed on (b, look_ahead, l, e)
		# (saves the lm_head matmul cost over n_thoughts) and then broadcast back over n for
		# downstream use. Targets are identical across n by construction.
		prior_hidden_per_b = prior_hidden_states[ :, 0 ].contiguous()
		prior_loss_per_b = self._flce_loss( prior_hidden_per_b, targets[ :, 0 ] )
		prior_cross_entropy_loss = prior_loss_per_b.unsqueeze( 1 ).expand( -1, self.n_thoughts, -1, -1 ).contiguous()
		prior_cross_entropy_loss = prior_cross_entropy_loss.masked_fill( mask_for_loss, t.nan )
		prior_cross_entropy_loss = x.reduce( "b n [d] l", prior_cross_entropy_loss, op = t.nanmean )

		# Posterior CE (for REINFORCE reward, not directly optimized)
		with t.no_grad():
			post_cross_entropy_loss = self._flce_loss( post_hidden_states, targets )
			post_cross_entropy_loss = post_cross_entropy_loss.masked_fill( mask_for_loss, t.nan )
			post_cross_entropy_loss = x.reduce( "b n [d] l", post_cross_entropy_loss, op = t.nanmean )

		# Thought CE (REINFORCE log-probabilities). Temperature scaling is applied to the hidden
		# state — equivalent to dividing logits by T since lm_head is linear.
		thought_targets = ts[ ..., 2:2 + self.thought_depth, : ]
		thought_ce = self._flce_loss( thought_hidden_states, thought_targets, scale = self.reinforce_temperature )
		thought_ce = thought_ce.masked_fill( x.rearrange("b l -> b 1 1 l", ~padding_mask.bool())[ ..., :-self.look_ahead ], t.nan )
		thought_ce = x.reduce( "b n [d] l", thought_ce, op = t.nanmean )

		# Compute soft-reward advantage (max-entropy RL)
		# Hard reward: how much thought improved prediction
		reinforce_reward_loss = mixed_cross_entropy_loss.detach() if self.gated_reinforce else post_cross_entropy_loss.detach()
		r = prior_cross_entropy_loss - reinforce_reward_loss
		# Soft reward: augment with thought log-probability to encourage exploration
		# thought_ce = -mean(log π(τ)), so adding it rewards higher-entropy thought sequences
		r_soft = r + self.coef_entropy * thought_ce.detach()
		r_soft_mean = x.mean( "b [n] l -> b 1 l", r_soft )
		advantage = t.nn.functional.relu( r_soft - r_soft_mean ).detach()

		# Apply REINFORCE loss
		thought_loss = thought_ce * advantage

		# Thought entropy was accumulated per-slice inside the generation loop (no_grad).
		thought_entropy = thought_entropy_acc.masked_fill(
			x.rearrange( "b l -> b 1 1 l", ~padding_mask.bool() )[ ..., :-self.look_ahead ], t.nan )
		thought_entropy = x.reduce( "b n [d] l", thought_entropy, op = t.nanmean )

		# # Compute confidence loss
		# conf_alpha = alpha[ ..., 1:, :-(self.look_ahead - 1), : ]
		# confidence = self.confidence_head(
		# 	prior_hidden_states[ ..., 1:, :-(self.look_ahead - 1), : ],
		# 	hidden_states[ ..., 1:, :-(self.look_ahead - 1), : ],
		# 	conf_alpha )
		# # Don't propagate externally through the mixer head
		# mixer_ref = conf_alpha.detach()
		# mixer_targets = alpha[ ..., 0, 1:, : ].unfold( -2, self.look_ahead - 1, 1 ).detach()
		# mixer_targets = x.rearrange( "b n l a d -> b n d l a", mixer_targets, a = 1 ).detach()
		#
		# confidence_targets = (mixer_targets.ge( mixer_ref )).to( dtype = confidence.dtype )
		# confidence_loss = t.nn.functional.binary_cross_entropy( confidence, confidence_targets, reduction = "none" )
		# confidence_loss = x.rearrange( "b n d l a -> b n d l", confidence_loss, a = 1 )
		# confidence_loss = confidence_loss.masked_fill( padding_mask[ ..., :-(2 * self.look_ahead - 1) ], t.nan )
		# confidence_loss = x.reduce( "b n [d] l", confidence_loss, op = "nanmean" )

		loss = (self.beta_mixed * mixed_cross_entropy_loss
			+ self.beta_thought * thought_loss + self.beta_stability * prior_cross_entropy_loss)
		loss = t.nanmean( loss )

		# Return raw per-sample tensors. Reduction to scalars (with .item()
		# syncs and python conditionals on tensor values) happens in
		# compute_stats_from_raw on the trainer side, keeping training_forward
		# pure-tensor and compile-friendly.
		raw = {
			"loss": loss,
			"mixed_cross_entropy_loss": mixed_cross_entropy_loss,
			"prior_cross_entropy_loss": prior_cross_entropy_loss,
			"post_cross_entropy_loss": post_cross_entropy_loss,
			"thought_loss": thought_loss,
			"alpha": alpha,
			"r": r,
			"advantage": advantage,
			"thought_entropy": thought_entropy,
		}

		return loss, None, raw

	@staticmethod
	def compute_stats_from_raw( raw ):
		"""Reduce raw per-sample tensors from training_forward into a dict of
		scalars suitable for self.log(). Lives outside the forward so the
		forward stays compile-friendly — every .item() and every python
		conditional on a tensor value is here, not in the compute graph."""
		mixed_ce = raw[ "mixed_cross_entropy_loss" ]
		prior_ce = raw[ "prior_cross_entropy_loss" ]
		post_ce = raw[ "post_cross_entropy_loss" ]
		thought_loss = raw[ "thought_loss" ]
		alpha = raw[ "alpha" ]
		r = raw[ "r" ]
		advantage = raw[ "advantage" ]
		thought_entropy = raw[ "thought_entropy" ]
		loss = raw[ "loss" ]

		with t.no_grad():
			pooled_alpha = x.reduce( "b n [d] l a", alpha, op = t.nanmean ).squeeze( -1 )
			post_advantage = prior_ce - post_ce
			valid_mask = ~(post_ce.isnan() | prior_ce.isnan())
			alpha_masked = pooled_alpha.where( valid_mask, t.tensor( float( "nan" ), device = pooled_alpha.device ) )
			adv_masked = post_advantage.where( valid_mask, t.tensor( float( "nan" ), device = post_advantage.device ) )
			weighted_adv_den = t.nansum( alpha_masked )
			weighted_post_advantage = (
				(t.nansum( alpha_masked * adv_masked ) / weighted_adv_den).item()
				if weighted_adv_den > 0 else float( "nan" )
			)

			return {
				"mixed_cross_entropy_avg": t.nanmean( mixed_ce ).item(),
				"mixed_cross_entropy_min": nanmin( mixed_ce ).item(),
				"mixed_cross_entropy_max": nanmax( mixed_ce ).item(),
				"prior_cross_entropy_avg": t.nanmean( prior_ce ).item(),
				"prior_cross_entropy_min": nanmin( prior_ce ).item(),
				"prior_cross_entropy_max": nanmax( prior_ce ).item(),
				"posterior_cross_entropy_avg": t.nanmean( post_ce ).item(),
				"posterior_cross_entropy_min": nanmin( post_ce ).item(),
				"posterior_cross_entropy_max": nanmax( post_ce ).item(),
				"thought_loss_avg": t.nanmean( thought_loss ).item(),
				"thought_loss_min": nanmin( thought_loss ).item(),
				"thought_loss_max": nanmax( thought_loss ).item(),
				"alpha_avg": t.nanmean( x.reduce( "b n [d] l a", alpha, op = t.nanmean ) ).item(),
				"alpha_min": nanmin( x.reduce( "b n [d] l a", alpha, op = t.nanmean ) ).item(),
				"alpha_max": nanmax( x.reduce( "b n [d] l a", alpha, op = t.nanmean ) ).item(),
				"r_avg": t.nanmean( r ).item(),
				"r_min": nanmin( r ).item(),
				"r_max": nanmax( r ).item(),
				"advantage_avg": t.nanmean( advantage ).item(),
				"advantage_min": nanmin( advantage ).item(),
				"advantage_max": nanmax( advantage ).item(),
				"thought_entropy_avg": t.nanmean( thought_entropy ).item(),
				"thought_entropy_min": nanmin( thought_entropy ).item(),
				"thought_entropy_max": nanmax( thought_entropy ).item(),
				"corr_post_prior_ce": nancorr( post_ce, prior_ce ).item(),
				"corr_alpha_post_advantage": nancorr( pooled_alpha, post_advantage ).item(),
				"alpha_weighted_post_advantage": weighted_post_advantage,
				"loss": loss.item()
			}

	@classmethod
	def truncate_cache( cls, kv_cache, l ):
		if isinstance( kv_cache, DynamicCache ):
			kv_cache.crop( l )
		else:
			raise NotImplementedError( "Only DynamicCache is supported" )

	@classmethod
	def broadcast_cache_batch( cls, kv_cache, n ):
		"""Replicate each batch entry n times (interleaved) along batch dim of cache."""
		if isinstance( kv_cache, DynamicCache ):
			for i in range( len( kv_cache.key_cache ) ):
				kv_cache.key_cache[ i ] = kv_cache.key_cache[ i ].repeat_interleave( n, dim = 0 )
				kv_cache.value_cache[ i ] = kv_cache.value_cache[ i ].repeat_interleave( n, dim = 0 )
		else:
			raise NotImplementedError( "Only DynamicCache is supported" )

	@t.inference_mode()
	def inference_forward(
			self, input_ids, padding_mask, thought_temperature, *, kv_cache = None, cache_pos = None,
			thought_depth = None ):
		if thought_depth is None:
			thought_depth = self.thought_depth
		b, l = input_ids.shape

		if kv_cache is not None:
			l += kv_cache.get_seq_length()

		if padding_mask is None:
			padding_mask = t.full( (b, l), 1, device = input_ids.device, dtype = input_ids.dtype )

		if cache_pos is None:
			seen = kv_cache.get_seq_length() if kv_cache is not None else 0
			cache_pos = t.arange( seen, l, device = input_ids.device )

		prior_logits, prior_hidden = self.naive_forward(
			input_ids, padding_mask, kv_cache = kv_cache, cache_pos = cache_pos
		)
		# Clone prior_hidden — the thought-generation loop's compiled calls
		# below would otherwise overwrite the CUDA Graph static buffer this
		# view points into before mixer_head reads it. prior_logits is fresh
		# (lm_head allocates its own output and is not part of the compiled
		# region) so it doesn't need cloning.
		prior_hidden = prior_hidden.clone()

		# Catenate the start token
		start_toks = t.full(
			(b, 1), self.start_thought_token_id, device = input_ids.device, dtype = input_ids.dtype )
		input_ids = t.cat( [ input_ids, start_toks ], dim = -1 ) if kv_cache is None else start_toks
		unpad = t.full( (b, 1), True, device = input_ids.device, dtype = input_ids.dtype )
		padding_mask = t.cat( [ padding_mask, unpad ], dim = -1 )
		cache_pos = t.cat( [ cache_pos, cache_pos[ ..., -1: ] + 1 ], dim = -1 ) if kv_cache is None else cache_pos[ ...,
																										 -1: ] + 1

		# Generate the thought
		for _ in range( thought_depth ):
			logits, _ = self.naive_forward(
				input_ids, padding_mask, kv_cache = kv_cache,
				cache_pos = cache_pos
			)

			toks = self.sample_thoughts( logits, thought_temperature )

			input_ids = t.cat( [ input_ids, toks ], dim = -1 ) if kv_cache is None else toks
			padding_mask = t.cat( [ padding_mask, unpad ], dim = -1 )
			cache_pos = t.cat( [ cache_pos, cache_pos[ ..., -1: ] + 1 ], dim = -1 ) if kv_cache is None else cache_pos[ ..., -1: ] + 1

		# Catenate the end token
		end_toks = t.full( (b, 1), self.end_thought_token_id, device = input_ids.device, dtype = input_ids.dtype )
		input_ids = t.cat( [ input_ids, end_toks ], dim = -1 )
		padding_mask = t.cat( [ padding_mask, unpad ], dim = -1 )

		cache_offset = cache_pos[ ..., -1: ] + t.arange( 2, device = cache_pos.device )
		cache_pos = t.cat( [ cache_pos, cache_offset[ -1: ] ], dim = -1 ) if kv_cache is None else cache_offset

		post_logits, post_hidden = self.naive_forward(
			input_ids, padding_mask, kv_cache = kv_cache, cache_pos = cache_pos
		)

		alpha = self.mixer_head( prior_hidden, post_hidden )

		if kv_cache is not None:
			self.truncate_cache( kv_cache, l )

		out = prior_logits * (1 - alpha) + post_logits * alpha
		assert out.shape == (b, 1, self.config.text_config.vocab_size)
		return out

	def forward(
			self,
			input_ids = None,
			attention_mask = None,
			position_ids = None,
			past_key_values = None,
			inputs_embeds = None,
			labels = None,
			use_cache = None,
			cache_position = None,
			output_attentions = None,
			output_hidden_states = None,
			return_dict = None,
			thought_temperature = None,
			thought_depth = None,
	):
		if thought_temperature is None:
			thought_temperature = self.thought_temperature
		if self.training:
			return self.training_forward(
				input_ids, labels, attention_mask, thought_temperature, kv_cache = past_key_values )
		else:
			logits = self.inference_forward(
				input_ids, attention_mask, thought_temperature, kv_cache = past_key_values, cache_pos = cache_position,
				thought_depth = thought_depth )
			return CausalLMOutput( logits = logits )


AutoModel.register( ThoughtModelConfig, ThoughtModel )

__all__ = [ "ThoughtModel", "ThoughtModelConfig", "MixerConfig", "WeightedMixerHead" ]
