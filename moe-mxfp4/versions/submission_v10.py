from task import input_t, output_t


_FUSED_MOE_PARAM_CACHE = None
_OUTPUT_CACHE = {}
_PAD_CACHE = {}


# Shape-aware policy inspired by mxfp4-mm exact-shape dispatch style:
# - EP-off (d_expert=256): preserve small-batch latency
# - EP-on  (d_expert=2048): push contiguous earlier
_SMALL_BS = 32
_EP_ON_CONTIGUOUS_THRESHOLD = 64
_EP_OFF_CONTIGUOUS_THRESHOLD = 128


def _get_fused_moe_params(fused_moe_fn):
    global _FUSED_MOE_PARAM_CACHE
    if _FUSED_MOE_PARAM_CACHE is None:
        import inspect

        _FUSED_MOE_PARAM_CACHE = set(inspect.signature(fused_moe_fn).parameters.keys())
    return _FUSED_MOE_PARAM_CACHE


def _get_output_buffer(m: int, h: int, device, dtype):
    key = (int(m), int(h), str(device), str(dtype))
    out = _OUTPUT_CACHE.get(key)
    if out is None:
        import torch

        out = torch.empty((m, h), device=device, dtype=dtype)
        _OUTPUT_CACHE[key] = out
    return out


def _get_pads(config):
    key = (
        int(config["d_hidden"]),
        int(config["d_hidden_pad"]),
        int(config["d_expert"]),
        int(config["d_expert_pad"]),
    )
    cached = _PAD_CACHE.get(key)
    if cached is None:
        cached = (
            config["d_hidden_pad"] - config["d_hidden"],
            config["d_expert_pad"] - config["d_expert"],
        )
        _PAD_CACHE[key] = cached
    return cached


def _maybe_contiguous(bs: int, d_expert: int, tensors):
    # keep very small batches untouched
    if bs <= _SMALL_BS:
        return tensors

    threshold = (
        _EP_ON_CONTIGUOUS_THRESHOLD if d_expert >= 2048 else _EP_OFF_CONTIGUOUS_THRESHOLD
    )
    if bs < threshold:
        return tensors

    converted = []
    for x in tensors:
        converted.append(x if x.is_contiguous() else x.contiguous())
    return tuple(converted)


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

    bs = int(hidden_states.shape[0])
    d_expert = int(config["d_expert"])
    (
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
    ) = _maybe_contiguous(
        bs,
        d_expert,
        (
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            gate_up_weight_scale_shuffled,
            down_weight_scale_shuffled,
            topk_weights,
            topk_ids,
        ),
    )

    hidden_pad, intermediate_pad = _get_pads(config)

    kwargs = {
        "expert_mask": None,
        "activation": ActivationType.Silu,
        "quant_type": QuantType.per_1x32,
        "doweight_stage1": False,
        "w1_scale": gate_up_weight_scale_shuffled,
        "w2_scale": down_weight_scale_shuffled,
        "a1_scale": None,
        "a2_scale": None,
        "hidden_pad": hidden_pad,
        "intermediate_pad": intermediate_pad,
    }

    params = _get_fused_moe_params(fused_moe)
    if "out" in params:
        kwargs["out"] = _get_output_buffer(
            hidden_states.shape[0],
            config["d_hidden"],
            hidden_states.device,
            hidden_states.dtype,
        )

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        **kwargs,
    )
