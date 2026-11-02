# Supertonic on ExecuTorch

This example exports [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)
to one dynamic ExecuTorch program with FP16 activations and the MLX or Vulkan
delegate.
The native runner supports one-shot or persistent text-to-speech synthesis
with MLX on Apple silicon or Vulkan on Linux. Vulkan export also supports
weight-only INT8 quantization with FP16 activations (W8A16).

Run all commands from the ExecuTorch repository root. Keep downloaded assets
and generated programs outside the source tree:

```bash
export SUPERTONIC_ASSETS="${TMPDIR:-/tmp}/supertonic-3"
export SUPERTONIC_PTE="${SUPERTONIC_ASSETS}/supertonic_fp16_mlx.pte"
```

## 1. Download the model assets

Install this example's Python dependencies, then download the exact reviewed
Hugging Face revision:

```bash
python -m pip install -r examples/models/supertonic/requirements.txt

hf download Supertone/supertonic-3 \
  --revision 3cadd1ee6394adea1bd021217a0e650ede09a323 \
  --local-dir "${SUPERTONIC_ASSETS}"
```

The export reads `onnx/tts.json` and the four ONNX models. The native runner
also reads `onnx/unicode_indexer.json` and one JSON file under `voice_styles/`.

## 2. Export the ExecuTorch program

```bash
python -m examples.models.supertonic.export.export_supertonic \
  --asset-dir "${SUPERTONIC_ASSETS}" \
  --output "${SUPERTONIC_PTE}" \
  --backend mlx \
  --max-text-length 512 \
  --max-latent-length 512 \
  --flow-steps 5
```

The generated PTE embeds the model weights; it does not require a `.ptd`
sidecar. Treat PTE files as trusted inputs: load only a PTE that you generated
or obtained from a trusted source.

### Vulkan and W8A16 export

To export on Linux or another platform with an ExecuTorch Python environment
that includes the Vulkan backend, select `--backend vulkan`. MLX is not
required for this path; omitting `--backend` preserves the MLX default.
Use an ExecuTorch installation built from the same checkout as the exporter.

```bash
export SUPERTONIC_VULKAN_PTE="${SUPERTONIC_ASSETS}/supertonic_w8a16_vulkan.pte"

python -m examples.models.supertonic.export.export_supertonic \
  --asset-dir "${SUPERTONIC_ASSETS}" \
  --output "${SUPERTONIC_VULKAN_PTE}" \
  --backend vulkan \
  --quantization w8 \
  --max-text-length 512 \
  --max-latent-length 512 \
  --flow-steps 5
```

Omit `--quantization w8`, or select `--quantization none`, to export an
unquantized FP16 Vulkan PTE. W8 is supported only with `--backend vulkan`;
the default backend and precision remain MLX and FP16.

W8 quantizes linear layers and equivalent ungrouped 1x1 convolution projections
using symmetric per-output-channel INT8 weights. Spatial and depthwise
convolutions, embeddings, normalization parameters, and scales remain
floating-point. Activations and method inputs/outputs stay FP16; this is not
W8A8. No external calibration dataset is required for weight-only quantization.

Both Vulkan modes preserve all four methods (`duration_predictor`,
`text_encoder`, `vector_estimator`, and `vocoder`), dynamic sequence bounds,
and the native pipeline metadata. Operators that cannot be delegated remain
in the ExecuTorch program and use CPU kernels linked by the native build.
Compare output quality and performance against the FP16 baseline when choosing
quantization for a deployment.

## 3. Build the native runner

For MLX on Apple silicon:

```bash
make supertonic-mlx
```

For Vulkan on Linux, install the ICU development package (`libicu-dev` on
Debian/Ubuntu) and the shader compiler described in the
[Vulkan development requirements](../../../docs/source/backends/vulkan/vulkan-overview.md#development-requirements).
Then build and select the matching PTE:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 make supertonic-vulkan
export SUPERTONIC_PTE="${SUPERTONIC_VULKAN_PTE}"
```

The runner is written to
`cmake-out/examples/models/supertonic/supertonic_runner`. The MLX build also
places the required `mlx.metallib` beside it; the Vulkan build does not use MLX.
Build the runner for the backend used during export.

## 4. Synthesize one WAV

```bash
./cmake-out/examples/models/supertonic/supertonic_runner \
  --pte="${SUPERTONIC_PTE}" \
  --asset_dir="${SUPERTONIC_ASSETS}" \
  --voice_style="${SUPERTONIC_ASSETS}/voice_styles/F1.json" \
  --text="Hello from Supertonic." \
  --language=en \
  --speed=1.05 \
  --seed=42 \
  --output="${SUPERTONIC_ASSETS}/hello.wav"
```

The runner writes a mono PCM16 WAV at the sample rate recorded in the model
metadata (44.1 kHz for the pinned assets).

## 5. Keep the runner warm

Use `--server_jsonl` to load the PTE once, perform one discarded warmup, and
process synthesis requests sequentially from stdin:

```bash
./cmake-out/examples/models/supertonic/supertonic_runner \
  --pte="${SUPERTONIC_PTE}" \
  --asset_dir="${SUPERTONIC_ASSETS}" \
  --voice_style="${SUPERTONIC_ASSETS}/voice_styles/F1.json" \
  --language=en \
  --speed=1.05 \
  --seed=42 \
  --server_jsonl
```

After loading and warmup, the runner emits this protocol-v1 `ready` schema;
`sample_rate` is always `44100` for the supported model:

```json
{"type":"ready","protocol_version":1,"sample_rate":44100,"load_seconds":0.03,"warmup_seconds":0.04}
```

Each request writes one complete WAV before returning its timing and RTF:

```json
{"type":"synthesize","id":1,"text":"Hello.","output":"/tmp/supertonic/1.wav"}
{"type":"result","id":1,"output":"/tmp/supertonic/1.wav","samples":82810,"audio_seconds":1.8778,"synthesis_seconds":0.0524,"rtf":0.0279}
```

Stdout is reserved for JSONL responses and flushed after every record;
diagnostics go to stderr. Request lines are limited to 64 KiB, and IDs must be
positive and strictly increasing. Output files are created atomically and must
not already exist. Malformed requests return an `error` record without stopping
the runner. Send `{"type":"shutdown"}` to receive `{"type":"stopped"}` and
exit cleanly; closing stdin also exits with status zero.

## 6. Run tests

The export tests cover backend selection, pointwise-convolution equivalence,
weight-only quantization, dynamic attention, serialized precision, and metadata:

```bash
python -m pytest \
  examples/models/supertonic/tests/test_export.py \
  examples/models/supertonic/tests/test_vulkan_quantization.py \
  examples/models/supertonic/tests/test_vulkan_pipeline.py
```

For native Vulkan tests, configure integration assets from the model directory:

```bash
cd examples/models/supertonic
cmake --preset supertonic-vulkan \
  -DSUPERTONIC_INTEGRATION_PTE="${SUPERTONIC_PTE}" \
  -DSUPERTONIC_INTEGRATION_ASSET_DIR="${SUPERTONIC_ASSETS}" \
  -DSUPERTONIC_INTEGRATION_STYLE="${SUPERTONIC_ASSETS}/voice_styles/F1.json"
ctest --preset supertonic-vulkan
```

This runs portable preprocessing/host tests, FP16/FP32 Vulkan buffer and texture
copy tests with dynamic resizing, one-shot synthesis, and the persistent JSONL
protocol. Synthesis integration tests skip when assets are not configured.
The Vulkan copy test requires a compatible Vulkan device.

## Platform and model limits

- Native MLX requires an Apple silicon Mac, macOS, Xcode command-line tools,
  CMake 3.26 or newer, and an ExecuTorch installation with the MLX backend
  and custom operations available.
- Native Vulkan requires Linux, CMake 3.26 or newer, ICU development headers,
  a compatible Vulkan device/driver, and an ExecuTorch installation built
  with Vulkan and the portable or optimized CPU kernels.
- Both native backends use FP16 activations. Weight-only INT8 is currently
  available for Vulkan export; the MLX workflow remains FP16.
- Exported programs use dynamic sequence lengths, five flow-matching steps,
  batch size 1, and exactly one voice style.
- The commands above export maximum text and latent lengths of 512. A single
  sentence is never split, so a sentence that exceeds the exported text bound
  is rejected.
- Language tags and voice styles are limited to those provided by the pinned
  Supertonic 3 release.

## Provenance and licensing

The Supertonic architecture and portions of this integration are adapted from
[Supertone's Supertonic repository](https://github.com/supertone-inc/supertonic)
at revision
[`7e2804f96016a7028cb1ed627353c61c1e9dd281`](https://github.com/supertone-inc/supertonic/commit/7e2804f96016a7028cb1ed627353c61c1e9dd281),
which is licensed under the MIT License. The complete upstream copyright and
license notice is preserved in [`NOTICE`](NOTICE). ExecuTorch-specific code is
licensed under the BSD-style license in the repository root.

Model weights, voice styles, configuration, and other assets downloaded from
[Hugging Face revision
`3cadd1ee6394adea1bd021217a0e650ede09a323`](https://huggingface.co/Supertone/supertonic-3/tree/3cadd1ee6394adea1bd021217a0e650ede09a323)
are separately licensed under the BigScience Open RAIL-M License included with
that release. This repository does not redistribute those assets. Do not add
the downloaded weights or an exported PTE containing them to the ExecuTorch
repository; obtain the assets from Hugging Face and review their license and use
restrictions before use or distribution.
