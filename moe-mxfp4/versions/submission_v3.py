from task import input_t, output_t


_CONTIGUOUS_THRESHOLD = 128


def custom_kernel(data: input_t) -> output_t:
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
