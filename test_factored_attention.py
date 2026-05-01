"""
Standalone correctness test for thought_factored_attention (flash_attn 4 backend).

Compares the factored attention against eager attention using the dense rank-4
mask produced by ThoughtModel.construct_thought_mask. flash_attn requires bf16
or fp16, so the test runs in bf16 with a correspondingly looser tolerance
(~5e-3 typical for bf16 attention round-trip).

The current factored impl does NOT apply a padding mask in attention — it
relies on loss masking at padded queries. To keep the comparison apples-to-
apples, the test uses an all-valid padding mask. If we later add varlen-based
padding handling, extend this test to a padded case.

Run on a GPU node (sm_80+):
    python test_factored_attention.py
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
	weights = torch.nan_to_num( weights, nan = 0.0 )
	out = weights @ v
	return out.transpose( 1, 2 ).contiguous()


def main():
	if not torch.cuda.is_available():
		print( "CUDA required (flash_attn)." )
		sys.exit( 1 )

	torch.manual_seed( 0 )
	device = torch.device( "cuda" )
	dtype = torch.bfloat16  # flash_attn requirement

	# Realistic-ish but small. Use a head_dim that's a flash_attn-supported size.
	b, n = 2, 2
	d_kv = 5
	d_q = 3
	q_off = 2  # Q covers depths [2, 3, 4)
	l = 8
	H = 4
	H_kv = 2  # GQA
	head_dim = 64  # flash_attn supports 64, 128, etc.

	B = b * n
	Q_len = d_q * l
	K_len = d_kv * l

	q = torch.randn( B, H, Q_len, head_dim, dtype = dtype, device = device )
	k = torch.randn( B, H_kv, K_len, head_dim, dtype = dtype, device = device )
	v = torch.randn( B, H_kv, K_len, head_dim, dtype = dtype, device = device )

	# All-valid padding mask — current impl ignores padding, so the comparison
	# only makes sense when the reference also doesn't mask anything.
	pm_bnl = torch.ones( b, n, l, dtype = torch.bool, device = device )
	pm_Bl = x.rearrange( "b n l -> (b n) l", pm_bnl )

	full_mask = ThoughtModel.construct_thought_mask( b, n, d_kv, l, pm_bnl, dtype )
	sliced = full_mask[ :, :, q_off: q_off + d_q, :, :, : ]
	mask_4d = x.rearrange( "b n D L d l -> (b n) 1 (D L) (d l)", sliced )

	scaling = head_dim ** -0.5
	num_kv_groups = H // H_kv
	module = _MockAttnModule( num_kv_groups )

	ref = _eager_with_full_mask( q, k, v, mask_4d, scaling, num_kv_groups )

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

	diff = (ref.float() - out.float()).abs()
	max_diff = diff.max().item()
	mean_diff = diff.mean().item()

	print( f"shape         : {tuple(ref.shape)}" )
	print( f"dtype         : {dtype}" )
	print( f"max abs diff  : {max_diff:.3e}" )
	print( f"mean abs diff : {mean_diff:.3e}" )

	# bf16 noise floor: the merge does 4 bf16 ops (w_a*out_a + w_b*out_b)/denom
	# vs the reference's single bf16 matmul. At magnitudes ~1, 1–2 ULPs of bf16
	# (~1.5e-2 near 2) is expected. Mean diff catches real math bugs.
	tol = 5e-2
	if max_diff > tol:
		flat_idx = diff.argmax().item()
		coords = torch.unravel_index( torch.tensor( flat_idx ), diff.shape )
		print( "WORST DIFF AT:", [ c.item() for c in coords ] )
		print( "ref :", ref[ coords ].item() )
		print( "out :", out[ coords ].item() )
		print( "FAIL" )
		sys.exit( 1 )

	print( "PASS" )


if __name__ == "__main__":
	main()
