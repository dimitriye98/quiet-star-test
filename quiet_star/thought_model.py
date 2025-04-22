import einx as x
import torch as t
from transformers import GenerationMixin, PreTrainedModel, DynamicCache, PretrainedConfig, AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput


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
		return self.mlp( catted_states )


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


class ThoughtModelConfig( PretrainedConfig ):
	model_type = "thought_model"
	is_composition = True
	sub_configs = { "mixer_config": MixerConfig, "text_config": AutoConfig }

	def __init__(
			self,
			*,
			beta_cross_entropy: float = 1.0,
			beta_thought: float = 1e6,
			end_thought_token_id: int = None,
			look_ahead: int = 4,
			look_ahead_pass: int = None,
			n_thoughts: int = 2,
			pad_token_id: int = None,
			start_thought_token_id: int = None,
			thought_depth: int = 12,
			thought_temperature: float = 0.0,

			mixer_config: dict | MixerConfig = None,
			text_config: dict | PretrainedConfig = None,
			**kwargs ):

		self.beta_cross_entropy = beta_cross_entropy
		self.beta_thought = beta_thought
		self.end_thought_token_id = end_thought_token_id
		self.look_ahead = look_ahead
		self.look_ahead_pass = look_ahead_pass
		self.n_thoughts = n_thoughts
		self.pad_token_id = pad_token_id
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

		self.beta_cross_entropy = config.beta_cross_entropy
		self.beta_thought = config.beta_thought
		self.look_ahead = config.look_ahead
		self.look_ahead_pass = config.look_ahead_pass
		self.mixer_config = config.mixer_config
		self.n_thoughts = config.n_thoughts
		self.thought_depth = config.thought_depth
		self.thought_temperature = config.thought_temperature

		self.lm_model = lm_model if lm_model is not None else AutoModel.from_config( config.text_config )
		self.mixer_head = WeightedMixerHead( config, lm_model.config.hidden_size )

		self.post_init()

	@staticmethod
	def construct_thought_mask( b, n, d, l, padding_mask, dtype ):
		device = padding_mask.device

		padding_mask = padding_mask.bool()

		range_d = x.rearrange( "d -> D L d l", t.arange( d, device = device ), l = l, d = d, D = d, L = l )
		range_D = x.rearrange( "D -> D L d l", t.arange( d, device = device ), l = l, d = d, D = d, L = l )
		range_l = x.rearrange( "l -> D L d l", t.arange( l, device = device ), l = l, d = d, D = d, L = l )
		range_L = x.rearrange( "L -> D L d l", t.arange( l, device = device ), l = l, d = d, D = d, L = l )

		# Allow attention only for tokens which precede in depth, or are in the initial input sequence and precede in time
		countercausal_mask = (range_D <= range_d) | ((range_d == 0) & (range_L <= range_l))
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

	def broadcast_logits( self, ts, kv_cache, layers_cached, layer_to_gen, mask, keep = 1 ):
		b, n, d, l = ts.shape

		if keep is None:
			keep = d

		if kv_cache is not None:
			cache_pos = x.rearrange( "d l -> (d l)", self.prepare_cache_positions( layers_cached, layer_to_gen, l ) )
		else:
			cache_pos = None

		a_mask = x.rearrange( "b n D L d l -> (b n) 1 (D L) (d l)", mask )
		input_ids = x.rearrange( "b n d l -> (b n) (d l)", ts )
		position_ids = x.rearrange(
			"d l -> (b n) (d l)", t.arange( l, device = ts.device ) + t.arange( d, device = ts.device ).unsqueeze( -1 ),
			b = b, n = n )

		out = self.lm_model(
			input_ids = input_ids,
			position_ids = position_ids,
			attention_mask = a_mask,
			past_key_values = kv_cache,
			use_cache = kv_cache is not None,
			output_attentions = False,
			output_hidden_states = True,
			return_dict = True,
			cache_position = cache_pos,
			num_logits_to_keep = l * keep,
		)

		log = x.rearrange( "(b n) (d l) v -> b n d l v", out.logits, b = b, n = n, d = keep, l = l )
		hidden = x.rearrange( "(b n) (d l) e -> b n d l e", out.hidden_states[ -1 ], b = b, n = n, d = d, l = l )[ ...,
		-keep:, :, : ]

		return log, hidden

	def naive_forward( self, input_ids, padding_mask, kv_cache = None, cache_pos = None, keep = 1 ):
		assert input_ids.shape[-1] == cache_pos.shape[-1]
		assert padding_mask.shape[-1] == input_ids.shape[-1] + kv_cache.get_seq_length()

		out = self.lm_model(
			input_ids = input_ids,
			attention_mask = padding_mask,
			past_key_values = kv_cache,
			use_cache = kv_cache is not None,
			output_attentions = False,
			output_hidden_states = True,
			return_dict = True,
			cache_position = cache_pos,
			num_logits_to_keep = keep,
		)

		log = out.logits
		hidden = out.hidden_states[ -1 ][ ..., -keep:, : ]

		return log, hidden

	def training_forward( self, input_ids, labels, padding_mask, thought_temperature, *, kv_cache ):
		assert kv_cache is not None
		assert input_ids.device == padding_mask.device

		# Truncate the input by look_ahead
		truncated_input = input_ids[ ..., :-self.look_ahead ]

		# Duplicate the batch for each thought
		ts = x.rearrange( "b l -> b n 1 l", truncated_input, n = self.n_thoughts )
		# padding_mask = x.rearrange( "b l -> b n l", padding_mask, n = self.n_thoughts )

		# Add <thought>
		start_toks = t.full(
			(ts.shape[ 0 ], ts.shape[ 1 ], 1, ts.shape[ 3 ]), self.start_thought_token_id, device = ts.device,
			dtype = ts.dtype )
		ts = t.cat( (start_toks, ts), dim = -2 )

		# rank-6 mask of shape (b, n, d, l, d, l)
		thought_mask = self.construct_thought_mask(
			b = ts.shape[ 0 ],
			n = ts.shape[ 1 ],
			d = 3 + self.look_ahead + self.thought_depth,  # input (1), start tok (1), depth, end tok (1), look ahead
			l = ts.shape[ 3 ],
			padding_mask = x.rearrange( "b l -> b n l", padding_mask, n = self.n_thoughts )[ ..., :-self.look_ahead ],
			dtype = self.dtype )

		# Generate thoughts
		layers_cached, layer_to_gen = 0, 2
		thought_logits = None
		for i in range( self.thought_depth ):
			slice_mask = thought_mask[ :, :, layers_cached: layer_to_gen, :, :, : ]

			new_logs, _ = self.broadcast_logits(
				ts[ :, :, layers_cached: layer_to_gen, : ],
				kv_cache,
				layers_cached,
				layer_to_gen,
				slice_mask )  # Discard hidden states
			if i == 0:
				thought_logits = new_logs
			else:
				thought_logits = t.cat( (thought_logits, new_logs), dim = -3 )

			new_toks = self.sample_thoughts( new_logs, thought_temperature ).detach()

			layers_cached = layer_to_gen
			layer_to_gen += 1

			ts = t.cat( (ts, new_toks), dim = -2 )

		# Add </thought>
		end_toks = t.full(
			(ts.shape[ 0 ], ts.shape[ 1 ], 1, ts.shape[ 3 ]), self.end_thought_token_id, device = ts.device,
			dtype = ts.dtype )
		ts = t.cat( (ts, end_toks), dim = -2 )
		layer_to_gen += 1

		# Prepare the sliding targets
		targets = labels.unfold( -1, self.look_ahead, 1 )[ :, 1: ].transpose( -1, -2 )
		targets = x.rearrange( "b d l -> b n d l", targets, n = self.n_thoughts, d = self.look_ahead )

		# Append the targets to the thoughts
		ts = t.cat( (ts, targets), dim = -2 )

		# Logits with thought
		if self.look_ahead_pass is not None:
			if self.look_ahead_pass != 1:
				raise NotImplementedError
			layer_to_gen += 1
			post_logits, post_hidden_states = None, None
			for i in range( 0, self.look_ahead ):
				slice_mask = thought_mask[ :, :, layers_cached: layer_to_gen, :, :, : ]
				log, hidden = self.broadcast_logits(
					ts[ :, :, layers_cached: layer_to_gen, : ],
					kv_cache,
					layers_cached,
					layer_to_gen,
					slice_mask,
					keep = 1 )
				if i == 0:
					post_logits = log
					post_hidden_states = hidden
				else:
					post_logits = t.cat( (post_logits, log), dim = -3 )
					post_hidden_states = t.cat( (post_hidden_states, hidden), dim = -3 )

				layers_cached = layer_to_gen
				layer_to_gen += 1
		else:
			layer_to_gen += self.look_ahead
			slice_mask = thought_mask[ :, :, layers_cached: layer_to_gen, :, :, : ]
			post_logits, post_hidden_states = self.broadcast_logits(
				ts[ :, :, layers_cached: layer_to_gen, : ],
				kv_cache,
				layers_cached,
				layer_to_gen,
				slice_mask,
				keep = self.look_ahead )
			# We won't use this anymore, but update it as a matter of hygiene
			layers_cached = layer_to_gen
			layer_to_gen += 1

		# Logits without thought
		prior_logits, prior_hidden_states = self.naive_forward(
			input_ids[ :, :-1 ], padding_mask[ :, :-1 ], keep = self.look_ahead )
		prior_logits = prior_logits.unfold( -2, self.look_ahead, 1 )
		# prior_logits = t.movedim( prior_logits, -1, -3 )  # b l v d -> b d l v
		prior_logits = x.rearrange( "b l v d -> b n d l v", prior_logits, n = self.n_thoughts )
		prior_hidden_states = prior_hidden_states.unfold( -2, self.look_ahead, 1 )
		# prior_hidden_states = t.movedim( prior_hidden_states, -1, -3 )  # b l e d -> b d l e
		prior_hidden_states = x.rearrange( "b l e d -> b n d l e", prior_hidden_states, n = self.n_thoughts )

		alpha = self.mixer_head( prior_hidden_states.expand_as( post_hidden_states ), post_hidden_states )
		logits = alpha * post_logits + (1 - alpha) * prior_logits

		# Compute cross entropy loss
		v = logits.shape[ -1 ]
		cross_entropy_loss = t.nn.functional.cross_entropy(
			logits.reshape( -1, v ), targets.reshape( -1 ),
			ignore_index = self.pad_token_id if self.pad_token_id is not None else -100,
			reduction = "none" ).reshape_as( targets )
		cross_entropy_loss = cross_entropy_loss.masked_fill(
			x.rearrange( "b l -> b 1 1 l", (~padding_mask.bool()) )[ ..., :-self.look_ahead ], t.nan )
		# Pool loss per thought
		cross_entropy_loss = x.reduce( "b n [d] l", cross_entropy_loss, op = t.nanmean )

		# Compute REINFORCE loss
		r = -cross_entropy_loss
		r_mean = x.mean( "b [n] l -> b 1 l", r )
		reward = t.nn.functional.relu( r - r_mean ).detach()

		# Apply REINFORCE loss
		thought_targets = ts[ ..., 2:2 + self.thought_depth, : ]
		v = thought_logits.shape[ -1 ]
		thought_loss = t.nn.functional.cross_entropy(
			thought_logits.reshape( -1, v ), thought_targets.reshape( -1 ),
			ignore_index = self.pad_token_id if self.pad_token_id is not None else -100,
			reduction = "none" ).reshape_as( thought_targets )
		thought_loss = thought_loss.masked_fill( (~padding_mask.bool())[ ..., :-self.look_ahead ], t.nan )
		# Pool loss per thought
		thought_loss = x.reduce( "b n [d] l", thought_loss, op = t.nanmean ) * reward

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

		loss = self.beta_cross_entropy * cross_entropy_loss + self.beta_thought * thought_loss
		loss = t.nanmean( loss )

		stats = {
			"cross_entropy_avg": t.nanmean( cross_entropy_loss ).item(),
			"cross_entropy_min": nanmin( cross_entropy_loss ).item(),
			"cross_entropy_max": nanmax( cross_entropy_loss ).item(),
			"thought_loss_avg": t.nanmean( thought_loss ).item(),
			"thought_loss_min": nanmin( thought_loss ).item(),
			"thought_loss_max": nanmax( thought_loss ).item(),
			"alpha_avg": t.nanmean( x.reduce( "b n [d] l a", alpha, op = t.nanmean ) ).item(),
			"alpha_min": nanmin( x.reduce( "b n [d] l a", alpha, op = t.nanmean ) ).item(),
			"alpha_max": nanmax( x.reduce( "b n [d] l a", alpha, op = t.nanmean ) ).item(),
			"r_avg": t.nanmean( r ).item(),
			"r_min": nanmin( r ).item(),
			"r_max": nanmax( r ).item(),
			"reward_avg": t.nanmean( reward ).item(),
			"reward_min": nanmin( reward ).item(),
			"reward_max": nanmax( reward ).item(),
			"loss": loss.item()
		}

		return loss, logits, stats

	@classmethod
	def truncate_cache( cls, kv_cache, l ):
		if isinstance( kv_cache, DynamicCache ):
			kv_cache.crop( l )
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

		prior_logits, prior_hidden = self.naive_forward(
			input_ids, padding_mask, kv_cache = kv_cache, cache_pos = cache_pos
		)

		# Catenate the start token
		start_toks = t.full(
			(b, 1), self.start_thought_token_id, device = input_ids.device, dtype = input_ids.dtype )
		input_ids = t.cat( [ input_ids, start_toks ], dim = -1 ) if kv_cache is None else start_toks
		unpad = t.full( (b, 1), True, device = input_ids.device, dtype = input_ids.dtype )
		padding_mask = t.cat( [ padding_mask, unpad ], dim = -1 )

		if cache_pos is not None:
			cache_pos = cache_pos[ ..., -1: ] + 1
			assert cache_pos.shape[-1] == 1
		elif kv_cache is not None:
			seen_tokens = kv_cache.get_seq_length()
			cache_pos = t.arange( seen_tokens, seen_tokens + input_ids.shape[ -1 ], device = input_ids.device )

		# Generate the thought
		for _ in range( thought_depth ):
			logits, _ = self.naive_forward(
				input_ids, padding_mask, kv_cache = kv_cache,
				cache_pos = cache_pos
			)

			toks = self.sample_thoughts( logits, thought_temperature )

			input_ids = t.cat( [ input_ids, toks ], dim = -1 ) if kv_cache is None else toks
			padding_mask = t.cat( [ padding_mask, unpad ], dim = -1 )
			cache_pos = cache_pos[ ..., -1: ] + 1

		# Catenate the end token
		end_toks = t.full( (b, 1), self.end_thought_token_id, device = input_ids.device, dtype = input_ids.dtype )
		input_ids = t.cat( [ input_ids, end_toks ], dim = -1 )
		padding_mask = t.cat( [ padding_mask, unpad ], dim = -1 )
		cache_pos = cache_pos[ ..., -1: ] + t.arange( 2, device = cache_pos.device )

		post_logits, post_hidden = self.naive_forward(
			input_ids, padding_mask, kv_cache = kv_cache, cache_pos = cache_pos
		)

		alpha = self.mixer_head( prior_hidden, post_hidden )

		self.truncate_cache( kv_cache, l )

		return prior_logits * (1 - alpha) + post_logits * alpha

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
