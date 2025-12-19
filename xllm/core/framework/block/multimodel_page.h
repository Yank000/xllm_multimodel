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
#include <torch/torch.h>

#include "platform/vmm_api.h"

namespace xllm {
class MultiModelPage {
 public:
  MultiModelPage(torch::Device device, int32_t page_id, int32_t page_size);

  ~MultiModelPage();

  const torch::Device& device() const { return device_; }

  PhyMemHandle get_phy_handle() const { return phy_handle_; }

  void init(size_t block_mem_size, size_t offset);

  void reset();

  size_t num_free_blocks();

  std::vector<int32_t> alloc(size_t num_blocks);

  void free(size_t block_id);

  bool full();

  bool empty();

  size_t get_page_id() const { return page_id_; }

  size_t get_page_size() const { return page_size_; }

  size_t get_offset() const { return offset_; }

 private:
  void _require_init();

  void set_block_range(size_t block_mem_size);

  bool _has_block(size_t block_id);

  bool _in_free_list(size_t block_id);

  torch::Device device_;
  PhyMemHandle phy_handle_;
  size_t page_size_;
  size_t page_id_;
  size_t start_block_;
  size_t end_block_;
  size_t num_kv_blocks_;
  std::vector<size_t> free_list_;

  size_t offset_;
};
}  // namespace xllm