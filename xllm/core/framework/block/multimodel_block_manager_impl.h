/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "block_manager.h"
#include "framework/kv_cache/kv_cache_event.h"
#include "multimodel_page.h"

namespace xllm {

class MultiModelBlockManagerImpl : public BlockManager {
 public:
  // allocate shared blocks when enable prefix cache
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) override {
    return {};
  }

  // cache blocks when enable prefix cache
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks) override {
    return;
  }
  void cache(const std::vector<Block>& blocks) override { return; }

  // get merged all dp rank KVCacheEvent
  void get_merged_kvcache_event(KvCacheEvent* event) const override { return; }

  Block allocate() override { return {}; }

  explicit MultiModelBlockManagerImpl(const Options& options);
  ~MultiModelBlockManagerImpl() /*{
    CHECK_EQ(num_free_blocks_, free_blocks_.size() - 1)
        << "Not all blocks have been freed";
  } */ = default;

  // Try to allocate blocks with num_blocks,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  void deallocate(const Slice<Block>& blocks) override;

  size_t num_blocks_in_prefix_cache() const override {
    if (options_.enable_prefix_cache()) {
      CHECK(prefix_cache_);
      return prefix_cache_->num_blocks();
    }
    return 0;
  }

  // free blocks num
  size_t num_free_blocks() const override { return num_free_blocks_; }

  // used blocks num
  size_t num_used_blocks() const override {
    if (options_.enable_prefix_cache()) {
      return num_used_blocks_;
    } else {
      return num_total_blocks() - num_free_blocks_;
    }
  }

  // current kv cache utilization.
  double kv_cache_utilization() const override {
    if (options_.enable_prefix_cache()) {
      return static_cast<double>(num_used_blocks_) / num_total_blocks();
    } else {
      return 1 - static_cast<double>(num_free_blocks_) / num_total_blocks();
    }
  }

  float get_gpu_cache_usage_perc() const override {
    return 1.0 - num_free_blocks_ * 1.0 / num_total_blocks();
  }

  // call BlockManager to free block used by Block.
  void free(int32_t block_id) override;

  // total blocks num
  size_t num_total_blocks() const override {
    return num_free_blocks_ + num_used_blocks_;
  }

 private:
  // check if has enough slots, if not, try to evict some blocks
  // from the prefix cache
  bool has_enough_blocks(uint32_t num_blocks);

 private:
  // prefix cache
  std::unique_ptr<PrefixCache> prefix_cache_;

  // reserved block id for padding
  // Block padding_block_;

  // number of used blocks
  std::atomic<size_t> num_used_blocks_{0};

  // number of used blocks
  std::atomic<size_t> num_free_blocks_{0};

  // block size
  size_t block_size_ = 0;

  // available page list
  std::unordered_map<int32_t, std::vector<std::unique_ptr<MultiModelPage>>>
      avail_pages_;

  // full page list
  std::unordered_map<int32_t, std::vector<std::unique_ptr<MultiModelPage>>>
      full_pages_;

  // page allocator
  std::shared_ptr<MultiModelPagePool> page_allocator_;

  // block memory size in each page
  size_t block_mem_size_ = 0;

  int32_t model_idx_ = 0;
};

}  // namespace xllm
