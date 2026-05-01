"""
Standalone correctness test for thought_factored_attention.

Compares the factored attention (Path A + Path B + LSE merge) against eager
attention with the dense rank-4 mask produced by ThoughtModel.construct_thought_mask.

Run on a GPU node:
    python test_factored_attention.py

Pure-PyTorch test in fp32; tolerance ~1e-5 should hold. Padded query positions
where ALL keys are masked have undefined output and are excluded from comparison.
"""
import sys

import torch
import einx as x

from quiet_star.thought_model import ThoughtModel
from quiet_star.factored_attention import thought_factored_attention


class _MockAttnModule:
	def __init__( self, num_kv_groups ):
		self.num_key_value_groups = num_kv_groups
		self.training = False


def _eager_with_full_mask( q, k, v, mask_4d, scaling, num_kv_groups ):
	"""Reference: full eager attention with the rank-4 additive mask.

	q: (B, H, Q, hd), k/v: (B, H_kv, K, hd), mask_4d: (B, 1, Q, K) additive.
	Returns (B, Q, H, hd) per HF convention.
	"""
	if num_kv_groups != 1:
		k = k.repeat_interleave( num_kv_groups, dim = 1 )
		v = v.repeat_interleave( num_kv_groups, dim = 1 )
	scores = (q @ k.transpose( -1, -2 )) * scaling  # (B, H, Q, K)
	scores = scores.float() + mask_4d.float()
	weights = torch.softmax( scores, dim = -1 ).to( q.dtype )
	# Replace NaN rows (fully masked) with zeros so we can still compare unmasked rows.
	weights = torch.nan_to_num( weights, nan = 0.0 )
	out = weights @ v
	return out.transpose( 1, 2 ).contiguous()


def _which_rows_have_any_key( mask_4d ):
	"""Return bool (B, Q): True iff that query row attends to at least one key.

	Used to skip rows where all keys are masked — softmax over -inf is undefined,
	and both impls just return zeros there, but the comparison itself is meaningless.
	"""
	# mask_4d is additive (-inf where masked, 0 where attended). Squeeze head dim.
	finite = (mask_4d.squeeze( 1 ) > -1e30)  # (B, Q, K)
	return finite.any( dim = -1 )  # (B, Q)


def main():
	torch.manual_seed( 0 )
	device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
	dtype = torch.float32  # fp32 for tight tolerance

	# Realistic-ish but small.
	b, n = 2, 2
	d_kv = 5
	d_q = 3
	q_off = 2  # Q covers depths [2, 3, 4)
	l = 4
	H = 4
	H_kv = 2  # GQA
	head_dim = 8

	B = b * n
	Q_len = d_q * l
	K_len = d_kv * l

	q = torch.randn( B, H, Q_len, head_dim, dtype = dtype, device = device )
	k = torch.randn( B, H_kv, K_len, head_dim, dtype = dtype, device = device )
	v = torch.randn( B, H_kv, K_len, head_dim, dtype = dtype, device = device )

	# Padding mask (b, l): mark last seq position as padded for batch 0.
	pm = torch.tensor(
		[ [ True, True, True, False ],
		  [ True, True, True, True ] ],
		dtype = torch.bool, device = device )  # (b, l)
	# construct_thought_mask wants (b, n, l)
	pm_bnl = pm.unsqueeze( 1 ).expand( -1, n, -1 ).contiguous()  # (b, n, l)
	# factored_attention wants (B, l) where B = b*n
	pm_Bl = x.rearrange( "b n l -> (b n) l", pm_bnl )

	# Build full d_kv mask, then slice Q to [q_off, q_off+d_q).
	full_mask = ThoughtModel.construct_thought_mask( b, n, d_kv, l, pm_bnl, dtype )
	# (b, n, D, L, d, l) where uppercase = Q. Slice Q-depth.
	sliced = full_mask[ :, :, q_off: q_off + d_q, :, :, : ]
	mask_4d = x.rearrange( "b n D L d l -> (b n) 1 (D L) (d l)", sliced )

	scaling = head_dim ** -0.5
	num_kv_groups = H // H_kv
	module = _MockAttnModule( num_kv_groups )

	# Reference
	ref = _eager_with_full_mask( q, k, v, mask_4d, scaling, num_kv_groups )

	# Factored
	out, _ = thought_factored_attention(
		module, q, k, v, attention_mask = None,
		scaling = scaling,
		padding_mask = pm_Bl,
		thought_d_q = d_q,
		thought_d_kv = d_kv,
		thought_l = l,
		q_depth_offset = q_off,
	)

	assert ref.shape == out.shape, f"shape mismatch ref={ref.shape} out={out.shape}"

	# Skip query rows where all keys are masked (output undefined).
	row_valid = _which_rows_have_any_key( mask_4d )  # (B, Q)
	row_valid_bcast = row_valid.unsqueeze( -1 ).unsqueeze( -1 ).expand_as( ref )

	diff = (ref - out).abs()
	masked_diff = torch.where( row_valid_bcast, diff, torch.zeros_like( diff ) )

	max_diff = masked_diff.max().item()
	mean_diff = masked_diff.mean().item()
	n_invalid_rows = int( (~row_valid).sum().item() )

	print( f"shape           : {tuple(ref.shape)}" )
	print( f"max abs diff    : {max_diff:.3e}" )
	print( f"mean abs diff   : {mean_diff:.3e}" )
	print( f"all-masked rows : {n_invalid_rows}/{B * Q_len}" )

	tol = 1e-5
	if max_diff > tol:
		# Print where the worst diff is for debugging.
		flat_idx = masked_diff.argmax().item()
		coords = torch.unravel_index( torch.tensor( flat_idx ), masked_diff.shape )
		print( "WORST DIFF AT:", [ c.item() for c in coords ] )
		print( "ref :", ref[ coords ].item() )
		print( "out :", out[ coords ].item() )
		print( "FAIL" )
		sys.exit( 1 )

	print( "PASS" )


if __name__ == "__main__":
	main()
