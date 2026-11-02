/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void test_copy(
    vkcompute::vkapi::ScalarType input_dtype,
    vkcompute::vkapi::ScalarType output_dtype,
    vkcompute::utils::StorageType storage_type) {
  using namespace vkcompute;
  using executorch::aten::Half;
  GraphConfig config;
  config.set_storage_type_override(storage_type);
  ComputeGraph graph(config);
  const std::vector<int64_t> maximum_shape{1, 3, 13};
  const IOValueRef input =
      graph.add_input_tensor(maximum_shape, input_dtype, storage_type);
  const ValueRef output =
      graph.add_tensor(maximum_shape, output_dtype, storage_type);
  const ValueRef none = graph.add_none();
  VK_GET_OP_FN("aten._to_copy.default")
  (graph, {input.value, none, none, none, none, none, none, output});
  const ValueRef staging_output = graph.set_output_tensor(output);
  graph.prepare();
  graph.prepack();

  for (const int64_t length : {13, 7, 1, 11}) {
    const std::vector<int64_t> shape{1, 3, length};
    std::vector<float> expected(3 * length);
    for (size_t index = 0; index < expected.size(); ++index) {
      expected[index] = static_cast<float>(index) * 0.25f - 3.0f;
    }
    graph.resize_input(0, shape);
    graph.propagate_resize();
    if (graph.sizes_of(output) != shape) {
      throw std::runtime_error("Vulkan copy did not propagate the input shape");
    }
    if (input_dtype == vkapi::kHalf) {
      std::vector<Half> half_input(expected.begin(), expected.end());
      graph.maybe_cast_and_copy_into_staging(
          input.staging, half_input.data(), half_input.size(), vkapi::kHalf);
    } else {
      graph.maybe_cast_and_copy_into_staging(
          input.staging, expected.data(), expected.size(), vkapi::kFloat);
    }
    graph.execute();
    std::vector<float> actual(expected.size());
    if (output_dtype == vkapi::kHalf) {
      std::vector<Half> half_output(expected.size());
      graph.maybe_cast_and_copy_from_staging(
          staging_output, half_output.data(), half_output.size(), vkapi::kHalf);
      for (size_t index = 0; index < actual.size(); ++index) {
        actual[index] = static_cast<float>(half_output[index]);
      }
    } else {
      graph.maybe_cast_and_copy_from_staging(
          staging_output, actual.data(), actual.size(), vkapi::kFloat);
    }
    for (size_t index = 0; index < expected.size(); ++index) {
      if (!std::isfinite(actual[index]) ||
          std::abs(actual[index] - expected[index]) > 1e-3f) {
        throw std::runtime_error("Vulkan copy returned incorrect values");
      }
    }
  }
}

} // namespace

int main() {
  try {
    for (const auto storage_type :
         {vkcompute::utils::kBuffer, vkcompute::utils::kTexture3D}) {
      for (const auto input_dtype :
           {vkcompute::vkapi::kFloat, vkcompute::vkapi::kHalf}) {
        for (const auto output_dtype :
             {vkcompute::vkapi::kFloat, vkcompute::vkapi::kHalf}) {
          test_copy(input_dtype, output_dtype, storage_type);
        }
      }
    }
    std::cout << "Supertonic Vulkan copy tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Supertonic Vulkan test failed: " << error.what() << '\n';
    return 1;
  }
}
