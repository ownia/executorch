# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from examples.models.supertonic.model.layers import Conv1dProjection, ConvNeXtBlock
from examples.models.supertonic.model.text_encoder import (
    FeedForward,
    RelativeMultiHeadAttention,
)
from examples.models.supertonic.model.vocoder import DecoderConvNeXtBlock, DecoderHead
from examples.models.supertonic.source_transformations.vulkan import (
    quantize_weight_only,
    replace_pointwise_convolutions,
    VulkanRelativeMultiHeadAttention,
)


@pytest.mark.parametrize("length", [1, 2, 4, 7])
def test_vulkan_relative_attention_matches_original_attention(length: int) -> None:
    attention = RelativeMultiHeadAttention(8, num_heads=2).half().eval()
    transformed = VulkanRelativeMultiHeadAttention.from_attention(attention)
    inputs = torch.randn(1, 8, length, dtype=torch.float16)
    mask = torch.ones(1, 1, length, length, dtype=torch.float16)

    with torch.no_grad():
        expected = attention(inputs, mask)
        actual = transformed(inputs, mask)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


from torch import nn
from torch.export import Dim, export


def test_pointwise_conversion_preserves_all_supertonic_projection_variants() -> None:
    models = (
        nn.ModuleDict(
            {
                "convnext": ConvNeXtBlock(8, kernel_size=3),
                "decoder_convnext": DecoderConvNeXtBlock(8, dilation=1, expansion=2),
                "attention": RelativeMultiHeadAttention(8, num_heads=2),
                "feed_forward": FeedForward(8, 16),
                "projection": Conv1dProjection(8, 16, 1),
                "spatial": Conv1dProjection(8, 16, 3),
                "decoder_head": DecoderHead(8, 16, 4),
            }
        )
        .half()
        .eval()
    )
    inputs = torch.randn(1, 8, 7, dtype=torch.float16)
    samples = {
        "convnext": (inputs,),
        "decoder_convnext": (inputs,),
        "attention": (inputs, torch.ones(1, 1, 7, 7, dtype=torch.float16)),
        "feed_forward": (inputs, torch.ones(1, 1, 7, dtype=torch.float16)),
        "projection": (inputs,),
        "spatial": (inputs,),
        "decoder_head": (inputs,),
    }
    with torch.no_grad():
        expected = {name: model(*samples[name]) for name, model in models.items()}
        assert replace_pointwise_convolutions(models) is models
        assert replace_pointwise_convolutions(models) is models
        actual = {name: model(*samples[name]) for name, model in models.items()}

    assert not any(
        isinstance(module, nn.Conv1d) and module.kernel_size == (1,)
        for module in models.modules()
    )
    assert isinstance(models["spatial"].net, nn.Conv1d)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("bias", [False, True])
def test_w8_quantization_preserves_fp16_activations_bias_and_dynamic_lengths(
    bias: bool,
) -> None:
    model = nn.Sequential(nn.Linear(8, 16, bias=bias)).half().eval()
    sample = torch.randn(1, 4, 8, dtype=torch.float16)
    shapes = ({1: Dim("text_length", min=1, max=4)},)
    program = export(model, (sample,), dynamic_shapes=shapes, strict=True)

    quantized = quantize_weight_only(
        program, (sample,), (sample[:, :2].contiguous(),), shapes
    )

    assert any(value.dtype == torch.int8 for value in quantized.state_dict.values())
    targets = {
        str(node.target) for node in quantized.graph.nodes if node.op == "call_function"
    }
    assert "quantized_decomposed.dequantize_per_channel.default" in targets
    assert "quantized_decomposed.quantize_per_tensor.default" not in targets
    for length in (1, 2, 4):
        inputs = sample[:, :length].contiguous()
        with torch.no_grad():
            actual = quantized.module()(inputs)
            expected = model(inputs)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-2)
