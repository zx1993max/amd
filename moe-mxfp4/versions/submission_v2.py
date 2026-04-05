from task import input_t, output_t


def _run_mxfp4_mm(data):
    import aiter
    import torch
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    a, b, b_q, b_shuffle, b_scale_sh = data
    with torch.inference_mode():
        a_q, a_scale = dynamic_mxfp4_quant(a.contiguous())
        a_q = a_q.view(dtypes.fp4x2)
        a_scale = e8m0_shuffle(a_scale).view(dtypes.fp8_e8m0)
        return aiter.gemm_a4w4(
            a_q,
            b_shuffle,
            a_scale,
            b_scale_sh,
            dtype=dtypes.bf16,
            bpreshuffle=True,
        )


def _run_moe_mxfp4(data):
    import torch
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    with torch.inference_mode():
        return fused_moe(
            hidden_states.contiguous(),
            gate_up_weight_shuffled.contiguous(),
            down_weight_shuffled.contiguous(),
            topk_weights.contiguous(),
            topk_ids.contiguous(),
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled.contiguous(),
            w2_scale=down_weight_scale_shuffled.contiguous(),
            a1_scale=None,
            a2_scale=None,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )


def custom_kernel(data: input_t) -> output_t:
    if len(data) == 5:
        return _run_mxfp4_mm(data)
    if len(data) == 12:
        return _run_moe_mxfp4(data)
    raise ValueError(f"Unsupported input tuple size: {len(data)}")
