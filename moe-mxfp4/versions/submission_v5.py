from task import input_t, output_t


_EXPERT_MASK_CACHE = {}


def _get_expert_mask(device, num_experts):
    key = (str(device), int(num_experts))
    mask = _EXPERT_MASK_CACHE.get(key)
    if mask is None:
        import torch

        mask = torch.ones((num_experts,), device=device, dtype=torch.bool)
        _EXPERT_MASK_CACHE[key] = mask
    return mask


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
    num_experts = config["n_routed_experts"] + config["n_shared_experts"]

    expert_mask = _get_expert_mask(hidden_states.device, num_experts)

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
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
