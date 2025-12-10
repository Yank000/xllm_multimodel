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
#include <map>

#include "block_multi_layer_xtensor.h"
#include "util/type_traits.h"

namespace xllm {

using BlockMultiLayerXTensorPairs =
    std::pair<std::vector<std::unique_ptr<BlockMultiLayerXTensor>>,
              std::vector<std::unique_ptr<BlockMultiLayerXTensor>>>;

class BlockMultiLayerXTensorTransfer {
 public:
  static BlockMultiLayerXTensorTransfer& get_instance() {
    static BlockMultiLayerXTensorTransfer instance;
    return instance;
  }

  void initialize(const std::vector<torch::Device>& devices);  // 没有用上

  void set_multi_layer_xtensor(
      std::vector<std::shared_ptr<XTensor>>& k_xtensors,
      std::vector<std::shared_ptr<XTensor>>& v_xtensors,
      torch::Device device);

  BlockMultiLayerXTensorPairs move_multi_layer_xtensors(int32_t device_id);

 private:
  BlockMultiLayerXTensorTransfer() = default;
  ~BlockMultiLayerXTensorTransfer() = default;
  DISALLOW_COPY_AND_ASSIGN(BlockMultiLayerXTensorTransfer);

 private:
  std::map<int32_t, std::vector<std::unique_ptr<BlockMultiLayerXTensor>>>
      multi_layer_k_xtensor_maps_;
  std::map<int32_t, std::vector<std::unique_ptr<BlockMultiLayerXTensor>>>
      multi_layer_v_xtensor_maps_;
};

}  // namespace xllm