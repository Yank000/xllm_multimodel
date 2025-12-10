/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>

#include "framework/xtensor/xtensor.h"

namespace xllm {
// for all layers
class BlockMultiLayerXTensor final {
 public:
  explicit BlockMultiLayerXTensor(
      std::vector<std::shared_ptr<XTensor>>& xtensors);

  ~BlockMultiLayerXTensor() = default;

  VirPtr get_block_vir_ptr(size_t offset, int64_t layer_idx) const {
    return xtensors_[layer_idx]->get_block_vir_ptr(offset);
  }

  int64_t get_num_layers() const { return num_layers_; }

 private:
  int64_t num_layers_ = 0;
  std::vector<std::shared_ptr<XTensor>> xtensors_;  // [num_layers]
};

}  // namespace xllm
