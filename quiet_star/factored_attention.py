"""
Factored attention for the Quiet-Star thought mask.

The mask construct_thought_mask builds is:
    ((depth_kv <= depth_q) & (seq_kv == seq_q)) | ((depth_kv == 0) & (seq_kv <= seq_q))
AND-ed with a padding mask on the K seq position.

That mask factorizes into two disjoint key sets per query at (depth_q, seq_q):
    A: depth_kv == 0, seq_kv <= seq_q          (depth-0 causal prefix)
    B: depth_kv in [1, depth_q], seq_kv == seq_q  (own column above depth-0)

We compute attention separately on each set, then merge via log-sum-exp.

Threading: thought_d_q, thought_d_kv, thought_l, q_depth_offset, padding_mask
arrive via **kwargs through HF's attention dispatch. broadcast_forward sets them
when the model is loaded with attn_implementation="thought_factored".

If the kwargs are absent (e.g. naive_forward / non-broadcast paths), we fall
through to standard SDPA so the same registered impl handles every call site.
"""

import torch
import einx as x
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def _eager_attention_with_lse( q, k, v, mask, scaling ):
	"""Eager attention returning (out, lse).

	q: (..., Q, head_dim), k/v: (..., K, head_dim), mask: (..., Q, K) bool or None.
	Mask convention: True = attend, False = masked out.
	Returns out in same dtype as q, lse in float32.
	"""
	scores = (q @ k.transpose( -1, -2 )) * scaling  # (..., Q, K)
	scores = scores.float()
	if mask is not None:
		scores = scores.masked_fill( ~mask, float( "-inf" ) )

	lse = torch.logsumexp( scores, dim = -1 )  # (..., Q)
	# logsumexp = -inf for fully-masked rows -> weights = exp(-inf - -inf) = NaN.
	# Replace those rows with zeros; the LSE merge will weight them to zero anyway.
	weights = (scores - lse.unsqueeze( -1 )).exp()
	weights = torch.nan_to_num( weights, nan = 0.0 )
	out = weights.to( v.dtype ) @ v  # (..., Q, head_dim)
	return out, lse


def _merge_lse( out_a, lse_a, out_b, lse_b ):
	"""Merge two attention outputs over disjoint key sets via log-sum-exp."""
	m = torch.maximum( lse_a, lse_b )  # (..., Q)
	# If both are -inf the row was fully masked everywhere -> output zero.
	all_masked = m.isneginf()
	m = torch.where( all_masked, torch.zeros_like( m ), m )
	w_a = (lse_a - m).exp()
	w_b = (lse_b - m).exp()
	denom = w_a + w_b
	denom = torch.where( denom == 0, torch.ones_like( denom ), denom )
	w_a_b = w_a.unsqueeze( -1 ).to( out_a.dtype )
	w_b_b = w_b.unsqueeze( -1 ).to( out_b.dtype )
	denom_b = denom.unsqueeze( -1 ).to( out_a.dtype )
	out = (w_a_b * out_a + w_b_b * out_b) / denom_b
	out = torch.where( all_masked.unsqueeze( -1 ), torch.zeros_like( out ), out )
	return out


def thought_factored_attention(
		module,
		query,           # (B, H, Q_len, head_dim)
		key,             # (B, H_kv, K_len, head_dim)
		value,           # (B, H_kv, K_len, head_dim)
		attention_mask,  # ignored in factored path; passed to fallback
		**kwargs,
):
	"""Custom attention implementation exploiting the Quiet-Star mask factorization.

	Returns (attn_output, None). attn_output is (B, Q_len, H, head_dim) per HF
	convention (the per-model wrapper does the final reshape to hidden).
	"""
	d_q = kwargs.pop( "thought_d_q", None )
	d_kv = kwargs.pop( "thought_d_kv", None )
	l = kwargs.pop( "thought_l", None )
	q_off = kwargs.pop( "q_depth_offset", None )
	padding_mask = kwargs.pop( "padding_mask", None )

	if d_q is None:
		# Non-broadcast call site (naive_forward, eval, etc.) — defer to SDPA.
		sdpa_fn = ALL_ATTENTION_FUNCTIONS[ "sdpa" ]
		return sdpa_fn( module, query, key, value, attention_mask, **kwargs )

	scaling = kwargs.get( "scaling" )
	if scaling is None:
		scaling = query.shape[ -1 ] ** -0.5

	B, H, Q_len, head_dim = query.shape
	_, H_kv, K_len, _ = key.shape
	assert Q_len == d_q * l, f"Q_len={Q_len} != d_q*l={d_q * l}"
	assert K_len == d_kv * l, f"K_len={K_len} != d_kv*l={d_kv * l}"

	# Repeat KV heads to match Q heads (GQA).
	if H_kv != H:
		repeat = H // H_kv
		key = key.repeat_interleave( repeat, dim = 1 )
		value = value.repeat_interleave( repeat, dim = 1 )

	# (B, H, d, l, head_dim)
	k_full = x.rearrange( "B H (d l) hd -> B H d l hd", key, d = d_kv, l = l )
	v_full = x.rearrange( "B H (d l) hd -> B H d l hd", value, d = d_kv, l = l )
	# Q stays flat in (d_q*l) for Path A; reshaped for Path B.

	# ============================================================
	# Path A: every query attends to depth-0 keys at seq_kv <= seq_q.
	# ============================================================
	q_a = query  # (B, H, d_q*l, head_dim)
	k_a = k_full[ :, :, 0 ]  # (B, H, l, head_dim)
	v_a = v_full[ :, :, 0 ]

	device = query.device
	arange_l = torch.arange( l, device = device )
	# seq position of each flat query index: depth d_idx, seq l_idx -> l_idx (independent of d_idx)
	seq_q_idx = arange_l.unsqueeze( 0 ).expand( d_q, -1 ).reshape( -1 )  # (d_q*l,)
	causal_a = arange_l.unsqueeze( 0 ) <= seq_q_idx.unsqueeze( -1 )  # (d_q*l, l)
	mask_a = causal_a.unsqueeze( 0 ).unsqueeze( 0 )  # (1, 1, d_q*l, l)
	if padding_mask is not None:
		# padding_mask: (B, l) bool, True = valid token (attend allowed)
		pm = padding_mask.bool().unsqueeze( 1 ).unsqueeze( 2 )  # (B, 1, 1, l)
		mask_a = mask_a & pm

	out_a, lse_a = _eager_attention_with_lse( q_a, k_a, v_a, mask_a, scaling )
	# out_a: (B, H, d_q*l, head_dim), lse_a: (B, H, d_q*l)

	# ============================================================
	# Path B: same-column attention to depth_kv in [1, depth_q].
	# Bring l up to batch dim so this becomes a tiny per-position attention along depth.
	# ============================================================
	if d_kv > 1:
		q_b = x.rearrange( "B H (d l) hd -> (B H l) d hd", query, d = d_q, l = l )
		k_b = x.rearrange( "B H d l hd -> (B H l) d hd", k_full[ :, :, 1: ] )
		v_b = x.rearrange( "B H d l hd -> (B H l) d hd", v_full[ :, :, 1: ] )

		# Causal in depth: Q at depth (q_off + i) attends to K at depth (1 + j) iff (1+j) <= (q_off+i).
		depth_q_abs = torch.arange( q_off, q_off + d_q, device = device )
		depth_k_abs = torch.arange( 1, d_kv, device = device )
		causal_b = depth_k_abs.unsqueeze( 0 ) <= depth_q_abs.unsqueeze( -1 )  # (d_q, d_kv-1)
		mask_b = causal_b.unsqueeze( 0 )  # (1, d_q, d_kv-1) — broadcasts over (B*H*l)

		if padding_mask is not None:
			# K at seq position p is masked iff that position is padded.
			# After "(B H l)" merge, dim 0 cycles fastest over l. Build a per-row K-mask of shape (B*H*l, 1, d_kv-1):
			# row index i = ((b*H) + h)*l + p, so the padding for row i depends on b and p.
			pm = padding_mask.bool()  # (B, l)
			pm_rows = pm.unsqueeze( 1 ).expand( -1, H, -1 ).reshape( -1 )  # (B*H*l,)
			# All d_kv-1 keys at this seq position are equally (in)valid.
			pm_b = pm_rows.unsqueeze( -1 ).unsqueeze( -1 )  # (B*H*l, 1, 1)
			mask_b = mask_b & pm_b

		out_b, lse_b = _eager_attention_with_lse( q_b, k_b, v_b, mask_b, scaling )
		out_b = x.rearrange( "(B H l) d hd -> B H (d l) hd", out_b, B = B, H = H, l = l )
		lse_b = x.rearrange( "(B H l) d -> B H (d l)", lse_b, B = B, H = H, l = l )
	else:
		# No keys above depth 0 -> Path B contributes nothing.
		out_b = torch.zeros_like( out_a )
		lse_b = torch.full_like( lse_a, float( "-inf" ) )

	out = _merge_lse( out_a, lse_a, out_b, lse_b )

	# HF expects (B, Q, H, head_dim).
	out = out.transpose( 1, 2 ).contiguous()
	return out, None


ALL_ATTENTION_FUNCTIONS.register( "thought_factored", thought_factored_attention )
