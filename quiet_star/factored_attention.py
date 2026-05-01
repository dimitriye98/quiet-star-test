"""
Factored attention for the Quiet-Star thought mask.

The mask construct_thought_mask builds is:
    ((depth_kv <= depth_q) & (seq_kv == seq_q)) | ((depth_kv == 0) & (seq_kv <= seq_q))
That factorizes into two disjoint key sets per query at (depth_q, seq_q):
    A: depth_kv == 0, seq_kv <= seq_q          (depth-0 causal prefix)
    B: depth_kv in [1, depth_q], seq_kv == seq_q  (own column above depth-0)

Path A is the dominant compute (length-l attention replicated across d_q depth
slots). Path B is a tiny per-(b, seq) attention along the depth axis with
seq dim ~10–15. The right tool differs:

  - Path A: flash_attn 4's flash_attn_func — fused kernel + native LSE.
            Reshape Q to (B*d_q, l, H, D), broadcast depth-0 K/V across depth
            slots, standard causal=True does within-depth seq causality.
  - Path B: eager (matmul + softmax + matmul). At ~14×14 per batch element,
            kernel launch overhead would dominate flash_attn; eager wins.
            Causal mask along depth axis is built explicitly to capture the
            q_off offset (q at depth q_off+i attends to k at depth 1+j iff
            j <= q_off + i - 1).
  - Merge: manual LSE merge over the two disjoint partitions.

Padding NOT applied in attention. The original SDPA path masked padded K
positions; here we rely on loss masking at padded queries. Add varlen-based
handling to Path A if loss diverges from the SDPA baseline.

Threading: thought_d_q, thought_d_kv, thought_l, q_depth_offset, padding_mask
arrive via **kwargs through HF's attention dispatch. broadcast_forward sets
them when the model is loaded with attn_implementation="thought_factored".

Non-broadcast call sites (naive_forward / eval) lack the thought kwargs and
fall through to SDPA so the same registered impl works everywhere.
"""

import torch
import einx as x
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from flash_attn.cute import flash_attn_func


def _eager_path_b( q, k, v, scaling, q_off ):
	"""Tiny per-(b, seq) attention along the depth axis. Returns (out, lse).

	q: (BL, d_q, H, head_dim)            in flash layout
	k, v: (BL, d_k, H_kv, head_dim)      where d_k = d_kv - 1
	Returns:
	    out: (BL, d_q, H, head_dim) in q.dtype
	    lse: (BL, H, d_q)           in float32
	"""
	_, d_q, H, _ = q.shape
	_, d_k, H_kv, _ = k.shape

	# GQA expansion (Path B sees the heads after our caller has already done
	# nothing to KV — we expand here for the eager matmul).
	if H_kv != H:
		repeat = H // H_kv
		k = k.repeat_interleave( repeat, dim = 2 )
		v = v.repeat_interleave( repeat, dim = 2 )

	# Move to PyTorch layout (BL, H, S, D) for clean matmul.
	q_p = q.transpose( 1, 2 )                  # (BL, H, d_q, D)
	k_p = k.transpose( 1, 2 )                  # (BL, H, d_k, D)
	v_p = v.transpose( 1, 2 )                  # (BL, H, d_k, D)

	scores = (q_p @ k_p.transpose( -1, -2 )) * scaling  # (BL, H, d_q, d_k)
	scores = scores.float()

	# Causal: Q at depth (q_off + i) attends K at depth (1 + j) iff j <= q_off + i - 1.
	device = q.device
	arange_q = torch.arange( d_q, device = device )
	arange_k = torch.arange( d_k, device = device )
	causal = arange_k.unsqueeze( 0 ) <= (arange_q.unsqueeze( -1 ) + (q_off - 1))  # (d_q, d_k)
	scores = scores.masked_fill( ~causal, float( "-inf" ) )

	lse = torch.logsumexp( scores, dim = -1 )  # (BL, H, d_q) fp32
	weights = (scores - lse.unsqueeze( -1 )).exp()
	weights = torch.nan_to_num( weights, nan = 0.0 )
	out = weights.to( v_p.dtype ) @ v_p        # (BL, H, d_q, D)
	out = out.transpose( 1, 2 ).contiguous()   # (BL, d_q, H, D)
	return out, lse


def _merge_lse( out_a, lse_a, out_b, lse_b ):
	"""Manual LSE merge over disjoint key partitions.

	out_a, out_b: (B, S, H, D) in same dtype.
	lse_a, lse_b: (B, H, S) in float32.
	Returns: out (B, S, H, D) in out_a.dtype.

	Weights are computed in fp32 then downcast to attention dtype before
	multiplication with the partial outputs — keeps per-token memory at
	(B, S, H, 1) instead of (B, S, H, D) fp32 copies of the partials.
	"""
	m = torch.maximum( lse_a, lse_b )                        # (B, H, S)
	all_masked = m.isneginf()
	m = torch.where( all_masked, torch.zeros_like( m ), m )
	w_a = (lse_a - m).exp()
	w_b = (lse_b - m).exp()
	denom = w_a + w_b
	denom = torch.where( denom == 0, torch.ones_like( denom ), denom )

	dt = out_a.dtype
	# (B, H, S) -> (B, S, H, 1) for broadcast over D.
	w_a_b = w_a.transpose( 1, 2 ).unsqueeze( -1 ).to( dt )
	w_b_b = w_b.transpose( 1, 2 ).unsqueeze( -1 ).to( dt )
	denom_b = denom.transpose( 1, 2 ).unsqueeze( -1 ).to( dt )
	out = (w_a_b * out_a + w_b_b * out_b) / denom_b
	out = torch.where(
		all_masked.transpose( 1, 2 ).unsqueeze( -1 ), torch.zeros_like( out ), out )
	return out


def thought_factored_attention(
		module,
		query,           # (B, H, Q_len, head_dim)
		key,             # (B, H_kv, K_len, head_dim)
		value,           # (B, H_kv, K_len, head_dim)
		attention_mask,  # ignored in factored path; passed to fallback
		**kwargs,
):
	d_q = kwargs.pop( "thought_d_q", None )
	d_kv = kwargs.pop( "thought_d_kv", None )
	l = kwargs.pop( "thought_l", None )
	q_off = kwargs.pop( "q_depth_offset", None )
	_padding_mask = kwargs.pop( "padding_mask", None )  # see module docstring

	if d_q is None:
		sdpa_fn = ALL_ATTENTION_FUNCTIONS[ "sdpa" ]
		return sdpa_fn( module, query, key, value, attention_mask, **kwargs )

	scaling = kwargs.get( "scaling" )
	if scaling is None:
		scaling = query.shape[ -1 ] ** -0.5

	# HF dispatches with (B, H, S, D); flash_attn wants (B, S, H, D).
	q = query.transpose( 1, 2 ).contiguous()
	k = key.transpose( 1, 2 ).contiguous()
	v = value.transpose( 1, 2 ).contiguous()

	B, Q_len, H, head_dim = q.shape
	_, K_len, H_kv, _ = k.shape
	assert Q_len == d_q * l, f"Q_len={Q_len} != d_q*l={d_q * l}"
	assert K_len == d_kv * l, f"K_len={K_len} != d_kv*l={d_kv * l}"

	# Expose depth dim on K/V.
	k_full = x.rearrange( "B (d l) H D -> B d l H D", k, d = d_kv, l = l )
	v_full = x.rearrange( "B (d l) H D -> B d l H D", v, d = d_kv, l = l )

	# ============================================================
	# Path A: every query attends to depth-0 keys at seq_kv <= seq_q.
	# ============================================================
	q_a = x.rearrange( "B (d l) H D -> (B d) l H D", q, d = d_q, l = l )
	# Broadcast (B, l, H_kv, D) -> (B*d_q, l, H_kv, D). expand+reshape forces a
	# contiguous copy; this is the main memory cost of Path A.
	k_a_base = k_full[ :, 0 ]
	v_a_base = v_full[ :, 0 ]
	k_a = k_a_base.unsqueeze( 1 ).expand( -1, d_q, -1, -1, -1 ).reshape( B * d_q, l, H_kv, head_dim ).contiguous()
	v_a = v_a_base.unsqueeze( 1 ).expand( -1, d_q, -1, -1, -1 ).reshape( B * d_q, l, H_kv, head_dim ).contiguous()

	out_a, lse_a = flash_attn_func(
		q_a, k_a, v_a,
		softmax_scale = scaling, causal = True, return_lse = True,
		# pack_gqa=None auto-selects the packed-GQA path, which hits a
		# crd2idx layout mismatch in store_LSE on flash_attn 4 beta.
		# Explicitly disable to use the unpacked head-replication path.
		pack_gqa = False )
	# out_a: (B*d_q, l, H, head_dim), lse_a: (B*d_q, H, l) fp32
	out_a = x.rearrange( "(B d) l H D -> B (d l) H D", out_a, B = B, d = d_q )
	lse_a = x.rearrange( "(B d) H l -> B H (d l)", lse_a, B = B, d = d_q )

	# ============================================================
	# Path B: queries at depth (q_off+i) attend keys at depth (1+j) for
	# j <= q_off + i - 1. Bring l up to batch dim and use eager (tiny attn).
	# ============================================================
	if d_kv > 1:
		q_b = x.rearrange( "B (d l) H D -> (B l) d H D", q, d = d_q, l = l )
		k_b = x.rearrange( "B d l H D -> (B l) d H D", k_full[ :, 1: ] )
		v_b = x.rearrange( "B d l H D -> (B l) d H D", v_full[ :, 1: ] )

		out_b, lse_b = _eager_path_b( q_b, k_b, v_b, scaling, q_off )
		out_b = x.rearrange( "(B l) d H D -> B (d l) H D", out_b, B = B, l = l )
		lse_b = x.rearrange( "(B l) H d -> B H (d l)", lse_b, B = B, l = l )

		out = _merge_lse( out_a, lse_a, out_b, lse_b )
	else:
		out = out_a

	# HF expects (B, Q, H, head_dim); flash_attn already returns that layout.
	return out, None


ALL_ATTENTION_FUNCTIONS.register( "thought_factored", thought_factored_attention )
