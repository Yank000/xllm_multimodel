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
#include <unordered_set>

#include "common/macros.h"
#include "multimodel/block_multi_layer_xtensor_transfer.h"
#include "multimodel_page.h"

namespace xllm {

// PhyPagePool is used to track memory pages of key and value. It is not
// thread safe. This class manages the allocation and deallocation of page.
class MultiModelPagePool final {
 public:
  struct Options {
    PROPERTY(uint64_t, num_total_pages) = 0;
    PROPERTY(int32_t, num_models) = 0;
  };

  MultiModelPagePool(const MultiModelPagePool::Options& options,
                     const torch::Device& device);

  void init();

  ~MultiModelPagePool() = default;

  // allocate pages for given block_mem_size, returns vector of pages (one per
  // layer)
  std::vector<std::unique_ptr<MultiModelPage>> allocate(int32_t model_idx);

  // get back pages to phy_page_pool
  void deallocate(std::vector<std::unique_ptr<MultiModelPage>> pages,
                  int32_t model_idx);

  int32_t get_page_id(int32_t block_id, size_t block_mem_size);
  /*
    void map(VirPtr vir_ptr, PhyMemHandle phy_handle) const;
    void map(VirPtr vir_ptr, uint32_t page_id, int64_t layer_idx) const;
    void batch_map(VirPtr vir_ptr,
                   std::vector<uint32_t>& page_ids,
                   uint32_t num_new_pages,
                   int64_t layer_idx) const;

    std::vector<uint32_t> get_page_id(int64_t block_id);

    // get num of total physical pages for key and value for all layers
    size_t get_num_total_phy_pages_per_layer() const {
      return free_phy_page_ids_.size();
    }

    // get num of free physical pages for key and value for one layer
    size_t get_num_free_phy_pages(int64_t pool_id) const {
      return num_free_phy_pages[pool_id];
    }

    // get num of used physical pages for key and value for one layer
    size_t get_num_used_phy_pages_per_layer() const {
      return free_phy_page_ids_.size() - num_free_phy_pages_per_layer_;
    }
  */
 private:
  DISALLOW_COPY_AND_ASSIGN(MultiModelPagePool);

 private:
  Options options_;

  torch::Device device_;

  // free physical pages
  std::vector<std::vector<std::vector<std::unique_ptr<MultiModelPage>>>>
      free_phy_pages_;  // [num_models, num_layers, num_pages]

  std::vector<std::atomic<uint32_t>>
      num_free_phy_pages_;  // [num_model_block_types]

  size_t page_size_;

  BlockMultiLayerXTensorPairs multi_layer_kv_xtensors_;

  void add_multi_layer_kv_xtensors(torch::Device device_);

  std::vector<int64_t> num_layers_;

  void map(VirPtr vir_ptr, PhyMemHandle phy_handle) const;
};
}  // namespace xllm
