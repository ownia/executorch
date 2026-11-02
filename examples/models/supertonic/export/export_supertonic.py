# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.export import export, ExportedProgram

from ..model.config import TTSConfig
from ..model.duration_predictor import DurationPredictor
from ..model.text_encoder import TextEncoder
from ..model.vector_estimator import VectorEstimator
from ..model.vocoder import Vocoder
from ..source_transformations.mlx import (
    exportable_vector_estimator,
    replace_relative_attention,
    replace_same_padding,
    replace_vocoder_causal_padding,
)
from ..source_transformations.vulkan import (
    quantize_weight_only,
    replace_pointwise_convolutions,
    VulkanRelativeMultiHeadAttention,
)

from . import common


_DEFAULT_EXPORT_BOUNDS = common.ExportBounds()


def _prepare_model(
    method_name: str,
    model: nn.Module,
    *,
    backend: str,
    quantization: str,
) -> nn.Module:
    model = replace_same_padding(model.eval())
    if quantization == "w8":
        model = replace_pointwise_convolutions(model)
    if method_name in ("duration_predictor", "text_encoder"):
        if not isinstance(model, (DurationPredictor, TextEncoder)):
            raise ValueError(f"{method_name} has an incompatible model type")
        if backend == "vulkan":
            model = replace_relative_attention(
                model, attention_type=VulkanRelativeMultiHeadAttention
            )
        else:
            model = replace_relative_attention(model)
    elif method_name == "vector_estimator":
        if not isinstance(model, VectorEstimator):
            raise ValueError("vector_estimator has an incompatible model type")
        model = exportable_vector_estimator(model)
    elif method_name == "vocoder":
        if not isinstance(model, Vocoder):
            raise ValueError("vocoder has an incompatible model type")
        model = replace_vocoder_causal_padding(model)
    else:
        raise ValueError(f"unsupported method {method_name!r}")
    return model


def export_programs(
    models: Mapping[str, nn.Module],
    config: TTSConfig,
    bounds: common.ExportBounds,
    *,
    flow_steps: int = common.DEFAULT_FLOW_STEPS,
    backend: str = "mlx",
    quantization: str = "none",
) -> dict[str, ExportedProgram]:
    common.validate_flow_steps(flow_steps)
    if backend not in ("mlx", "vulkan"):
        raise ValueError(f"unsupported backend {backend!r}")
    if quantization not in ("none", "w8"):
        raise ValueError(f"unsupported quantization {quantization!r}")
    if quantization == "w8" and backend != "vulkan":
        raise ValueError("w8 quantization requires the Vulkan backend")
    expected_methods = set(common.METHOD_NAMES)
    if set(models) != expected_methods:
        missing = sorted(expected_methods - set(models))
        extra = sorted(set(models) - expected_methods)
        raise ValueError(
            f"models must contain the exact method set; missing={missing}, extra={extra}"
        )
    samples = common.example_inputs(config, bounds, flow_steps=flow_steps)
    calibration_samples = (
        common.example_inputs(
            config,
            common.ExportBounds(
                text_max=min(bounds.text_max, 4),
                latent_max=min(bounds.latent_max, 4),
            ),
            flow_steps=flow_steps,
        )
        if quantization == "w8"
        else samples
    )
    shapes = common.dynamic_shapes(bounds)
    programs = {}
    with torch.no_grad():
        for method_name in common.METHOD_NAMES:
            model = _prepare_model(
                method_name,
                models[method_name],
                backend=backend,
                quantization=quantization,
            )
            programs[method_name] = export(
                model,
                samples[method_name],
                dynamic_shapes=shapes[method_name],
                strict=True,
            )
            if quantization == "w8":
                programs[method_name] = quantize_weight_only(
                    programs[method_name],
                    samples[method_name],
                    calibration_samples[method_name],
                    shapes[method_name],
                )
    return programs


def lower_to_mlx(
    programs: Mapping[str, ExportedProgram],
    metadata: Mapping[str, object],
):
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    return to_edge_transform_and_lower(
        dict(programs),
        transform_passes=get_default_passes(),
        partitioner={method_name: [MLXPartitioner()] for method_name in programs},
        # MLX lowers model buffers such as the vector-estimator frequencies as
        # delegate constants. Edge validation and dim-order rewriting currently
        # look them up as graph state and fail before partitioning.
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=dict(metadata),
    )


def lower_to_vulkan(
    programs: Mapping[str, ExportedProgram],
    metadata: Mapping[str, object],
):
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    return to_edge_transform_and_lower(
        dict(programs),
        partitioner={
            method_name: [
                VulkanPartitioner(
                    compile_options={
                        "require_dynamic_shapes": True,
                        "force_fp16": True,
                    }
                )
            ]
            for method_name in programs
        },
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
        constant_methods=dict(metadata),
    )


def to_executorch(edge_program):
    from executorch.exir import ExecutorchBackendConfig

    return edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )


def export_from_assets(
    asset_dir: str | Path,
    output_path: str | Path,
    *,
    backend: str = "mlx",
    quantization: str = "none",
    bounds: common.ExportBounds = _DEFAULT_EXPORT_BOUNDS,
    flow_steps: int = common.DEFAULT_FLOW_STEPS,
) -> Path:
    if backend == "mlx":
        lower = lower_to_mlx
    elif backend == "vulkan":
        lower = lower_to_vulkan
    else:
        raise ValueError(f"unsupported backend {backend!r}; expected 'mlx' or 'vulkan'")
    if quantization not in ("none", "w8"):
        raise ValueError(f"unsupported quantization {quantization!r}")
    if quantization == "w8" and backend != "vulkan":
        raise ValueError("w8 quantization requires the Vulkan backend")

    config, models = common.load_models(asset_dir)
    vocabulary_size = common.text_vocabulary_size(models)
    common.convert_models_to_fp16(models)
    programs = export_programs(
        models,
        config,
        bounds,
        flow_steps=flow_steps,
        backend=backend,
        quantization=quantization,
    )
    edge_program = lower(
        programs,
        common.runtime_metadata(
            config,
            bounds,
            text_vocabulary_size=vocabulary_size,
            flow_steps=flow_steps,
        ),
    )
    return common.save_pte(to_executorch(edge_program), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one dynamic FP16 Supertonic artifact."
    )
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("mlx", "vulkan"), default="mlx")
    parser.add_argument("--quantization", choices=("none", "w8"), default="none")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-latent-length", type=int, default=512)
    parser.add_argument(
        "--flow-steps",
        type=int,
        choices=(common.DEFAULT_FLOW_STEPS,),
        default=common.DEFAULT_FLOW_STEPS,
    )
    args = parser.parse_args()
    bounds = common.ExportBounds(
        text_max=args.max_text_length,
        latent_max=args.max_latent_length,
    )
    export_from_assets(
        args.asset_dir,
        args.output,
        backend=args.backend,
        quantization=args.quantization,
        bounds=bounds,
        flow_steps=args.flow_steps,
    )


if __name__ == "__main__":
    main()
