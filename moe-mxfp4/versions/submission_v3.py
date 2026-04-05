from task import input_t, output_t


_CONTIGUOUS_THRESHOLD = 128


def _run_mxfp4_mm(data):
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    a, b, b_q, b_shuffle, b_scale_sh = data
    m = a.shape[0]

    if m >= _CONTIGUOUS_THRESHOLD:
        a = a.contiguous()
        b_shuffle = b_shuffle.contiguous()
        b_scale_sh = b_scale_sh.contiguous()

    a_q, a_scale = dynamic_mxfp4_quant(a)
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

    bs = hidden_states.shape[0]
    if bs >= _CONTIGUOUS_THRESHOLD:
        hidden_states = hidden_states.contiguous()
        gate_up_weight_shuffled = gate_up_weight_shuffled.contiguous()
        down_weight_shuffled = down_weight_shuffled.contiguous()
        gate_up_weight_scale_shuffled = gate_up_weight_scale_shuffled.contiguous()
        down_weight_scale_shuffled = down_weight_scale_shuffled.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
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
