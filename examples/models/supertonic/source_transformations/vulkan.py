# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.export import export, ExportedProgram

from ..model.layers import Conv1dProjection, ConvNeXtBlock
from ..model.text_encoder import FeedForward, RelativeMultiHeadAttention
from ..model.vocoder import DecoderConvNeXtBlock, DecoderHead
from .mlx import MLXRelativeMultiHeadAttention


class VulkanRelativeMultiHeadAttention(MLXRelativeMultiHeadAttention):
    """Keep relative-attention dimensions explicit across Vulkan partitions."""

    @staticmethod
    def _relative_to_absolute(inputs: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = inputs.shape
        positions = torch.arange(length, device=inputs.device)
        indices = positions.unsqueeze(0) - positions.unsqueeze(1) + length - 1
        return torch.gather(
            inputs,
            -1,
            indices.reshape(1, 1, length, length).expand(batch, heads, length, length),
        )

    @staticmethod
    def _absolute_to_relative(inputs: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = inputs.shape
        queries = torch.arange(length, device=inputs.device)
        relative = torch.arange(2 * length - 1, device=inputs.device)
        indices = relative.unsqueeze(0) + queries.unsqueeze(1) - (length - 1)
        valid = (indices >= 0) & (indices < length)
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
        gathered = torch.gather(
            inputs,
            -1,
            safe_indices.reshape(1, 1, length, 2 * length - 1).expand(
                batch, heads, length, 2 * length - 1
            ),
        )
        return gathered * valid.reshape(1, 1, length, 2 * length - 1)


class _PointwiseConv1d(nn.Module):
    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        self.linear = nn.Linear(
            conv.in_channels,
            conv.out_channels,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.squeeze(-1))
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.transpose(1, 2)).transpose(1, 2)


def _as_linear(module: nn.Module) -> nn.Module:
    if (
        isinstance(module, nn.Conv1d)
        and module.kernel_size == (1,)
        and module.stride == (1,)
        and module.padding == (0,)
        and module.groups == 1
    ):
        return _PointwiseConv1d(module)
    return module


def replace_pointwise_convolutions(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (ConvNeXtBlock, DecoderConvNeXtBlock)):
            module.pwconv1 = _as_linear(module.pwconv1)
            module.pwconv2 = _as_linear(module.pwconv2)
        elif isinstance(
            module, (RelativeMultiHeadAttention, MLXRelativeMultiHeadAttention)
        ):
            module.conv_q = _as_linear(module.conv_q)
            module.conv_k = _as_linear(module.conv_k)
            module.conv_v = _as_linear(module.conv_v)
            module.conv_o = _as_linear(module.conv_o)
        elif isinstance(module, FeedForward):
            module.conv_1 = _as_linear(module.conv_1)
            module.conv_2 = _as_linear(module.conv_2)
        elif isinstance(module, Conv1dProjection):
            module.net = _as_linear(module.net)
        elif isinstance(module, DecoderHead):
            module.layer2 = _as_linear(module.layer2)
    return model


def quantize_weight_only(
    program: ExportedProgram,
    example_inputs: tuple[torch.Tensor, ...],
    calibration_inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: tuple[dict | None, ...],
) -> ExportedProgram:
    from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
        get_symmetric_quantization_config,
        VulkanQuantizer,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = VulkanQuantizer().set_global(
        get_symmetric_quantization_config(is_dynamic=False, weight_bits=8)
    )
    model = program.module()
    # Noncontiguous linear inputs otherwise lower to an unfused weight dequantizer.
    for node in list(model.graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        inputs = node.args[0]
        assert isinstance(inputs, torch.fx.Node)
        with model.graph.inserting_before(node):
            contiguous = model.graph.call_function(
                torch.ops.aten.clone.default,
                (inputs,),
                {"memory_format": torch.contiguous_format},
            )
        contiguous.meta = inputs.meta.copy()
        tensor = inputs.meta["val"]
        assert isinstance(tensor, torch.Tensor)
        contiguous.meta["val"] = tensor.clone(memory_format=torch.contiguous_format)
        node.replace_input_with(inputs, contiguous)
    model.recompile()
    prepared = prepare_pt2e(model, quantizer)
    with torch.no_grad():
        prepared(*calibration_inputs)
    converted = convert_pt2e(prepared)
    return export(
        converted,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
