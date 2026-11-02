# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from examples.models.supertonic.export import common, export_supertonic
from examples.models.supertonic.model.text_encoder import RelativeMultiHeadAttention
from examples.models.supertonic.source_transformations.vulkan import (
    quantize_weight_only,
    VulkanRelativeMultiHeadAttention,
)
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkDataType,
    VkTensor,
)
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)
from executorch.exir._serialize._program import (
    _extract_delegate_payload,
    deserialize_pte_binary,
)
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import Bool, Int, String, Tensor, TensorShapeDynamism
from torch import nn
from torch.export import Dim, export


class _Stage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(inputs.transpose(1, 2)))


@pytest.fixture(scope="module", params=("none", "w8"))
def serialized(request, tmp_path_factory):
    programs = {}
    for name in common.METHOD_NAMES:
        inputs = (torch.ones(1, 4, 4, dtype=torch.float16),)
        shapes = ({2: Dim(f"{name}_length", min=1, max=4)},)
        program = export(
            _Stage().half().eval(),
            inputs,
            dynamic_shapes=shapes,
            strict=True,
        )
        if request.param == "w8":
            program = quantize_weight_only(program, inputs, inputs, shapes)
        programs[name] = program
    metadata = {
        "get_sample_rate": 44100,
        "get_max_text_length": 4,
        "get_activation_dtype": "float16",
        "enable_dynamic_shape": True,
    }
    edge = export_supertonic.lower_to_vulkan(programs, metadata)
    et_program = export_supertonic.to_executorch(edge)
    assert not et_program._tensor_data
    directory = tmp_path_factory.mktemp("supertonic-vulkan")
    pte_path = common.save_pte(et_program, directory / "supertonic_fp16_vulkan.pte")
    assert set(directory.iterdir()) == {pte_path}
    data = pte_path.read_bytes()
    return deserialize_pte_binary(data).program, metadata, request.param, data


def test_all_four_methods_are_serialized_with_vulkan_delegates(serialized) -> None:
    program, metadata, _, _ = serialized
    plans = {plan.name: plan for plan in program.execution_plan}

    assert set(plans) == set(common.METHOD_NAMES) | set(metadata)
    for name in common.METHOD_NAMES:
        assert plans[name].delegates
        assert all(delegate.id == "VulkanBackend" for delegate in plans[name].delegates)


def test_vulkan_export_preserves_dynamic_fp16_inputs_and_outputs(serialized) -> None:
    program, _, _, _ = serialized

    for plan in program.execution_plan:
        if plan.name not in common.METHOD_NAMES:
            continue
        for index in (*plan.inputs, *plan.outputs):
            tensor = plan.values[index].val
            assert isinstance(tensor, Tensor)
            assert tensor.scalar_type == ScalarType.HALF
            assert tensor.shape_dynamism == TensorShapeDynamism.DYNAMIC_BOUND
            assert tensor.sizes == [1, 4, 4]


def test_vulkan_export_preserves_typed_constant_methods(serialized) -> None:
    program, metadata, _, _ = serialized
    plans = {plan.name: plan for plan in program.execution_plan}

    for name, expected in metadata.items():
        plan = plans[name]
        assert not plan.delegates
        value = plan.values[plan.outputs[0]].val
        if isinstance(value, Bool):
            actual = value.bool_val
        elif isinstance(value, Int):
            actual = value.int_val
        else:
            assert isinstance(value, String)
            actual = value.string_val
        assert actual == expected
        assert type(actual) is type(expected)


def test_vulkan_weight_precision_matches_quantization_mode(serialized) -> None:
    _, _, quantization, data = serialized

    for index in range(len(common.METHOD_NAMES)):
        payload = _extract_delegate_payload(data, "vulkan", delegate_index=index)
        assert payload is not None
        graph = flatbuffer_to_vk_graph(extract_vk_flatbuffer(payload))
        int8_weights = [
            value.value
            for value in graph.values
            if isinstance(value.value, VkTensor)
            and value.value.constant_id >= 0
            and value.value.datatype == VkDataType.INT8
        ]
        quantized_ops = [
            op for op in graph.chain if op.name == "aten._weight_int8pack_mm.default"
        ]
        if quantization == "w8":
            assert int8_weights
            assert len(quantized_ops) == 1
        else:
            assert not int8_weights
            assert not quantized_ops


def test_vulkan_relative_attention_has_consistent_dynamic_broadcast_bounds() -> None:
    length = Dim("text_length", min=1, max=16)
    attention = VulkanRelativeMultiHeadAttention.from_attention(
        RelativeMultiHeadAttention(8, num_heads=4).half().eval()
    ).eval()
    program = export(
        attention,
        (
            torch.randn(1, 8, 16, dtype=torch.float16),
            torch.ones(1, 1, 16, 16, dtype=torch.float16),
        ),
        dynamic_shapes=({2: length}, {2: length, 3: length}),
        strict=True,
    )
    edge = export_supertonic.lower_to_vulkan({"attention": program}, {})
    delegates = sum(
        "executorch_call_delegate" in str(node.target)
        for node in edge.exported_program("attention").graph.nodes
    )
    assert delegates > 0
    data = export_supertonic.to_executorch(edge).buffer
    checked = 0
    for index in range(delegates):
        payload = _extract_delegate_payload(data, "vulkan", delegate_index=index)
        assert payload is not None
        graph = flatbuffer_to_vk_graph(extract_vk_flatbuffer(payload))
        for op in graph.chain:
            if op.name != "aten.add.Tensor":
                continue
            left, right, out = (
                graph.values[arg].value for arg in (op.args[0], op.args[1], op.args[-1])
            )
            if not all(isinstance(value, VkTensor) for value in (left, right, out)):
                continue
            assert out.dims == list(torch.broadcast_shapes(left.dims, right.dims))
            checked += 1
    assert checked > 0
