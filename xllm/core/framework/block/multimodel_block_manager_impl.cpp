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

#include "multimodel_block_manager_impl.h"

#include <algorithm>

#include "framework/prefix_cache/prefix_cache_factory.h"
namespace xllm {

MultiModelBlockManagerImpl::MultiModelBlockManagerImpl(const Options& options)
    : BlockManager(options) {
  CHECK_GT(options.num_blocks(), 0) << "No blocks to allocate";
  CHECK_GT(options.block_size(), 0) << "Block size must be positive";
  if (options_.enable_prefix_cache()) {
    prefix_cache_ = create_prefix_cache(options.block_size(),
                                        options.enable_cache_upload());
    CHECK(prefix_cache_) << "Failed to create prefix cache!";
  }
  // num_pages:初始化预分配的物理页数量；kv_token_size_:每个token的kv向量大小
  size_t init_pages = options_.init_pages();
  block_size_ = options_.block_size();
  block_mem_size_ = block_size_ * options_.slot_size() /
                    2;  // slot_size包含了K和V,但是单个page只存储K或V
  model_idx_ = options_.model_idx();
  page_allocator_ = options_.multi_model_page_pools()[options_.devices()[0]];

  add_multi_layer_kv_xtensor(options_.devices()[0]);
  // TODO: refactor parallel
  int32_t page_size_ = 2 * 1024 * 1024;
  // TODO: refactor page_size config
  size_t max_page_num = options_.num_blocks() * block_mem_size_ / page_size_;

  offsets_.reserve(max_page_num);

  for (size_t i = 0; i < max_page_num; ++i) {
    offsets_.push_back(i * page_size_);
  }

  // refactor devices
  for (int32_t i = 0; i < init_pages; ++i) {
    std::vector<std::unique_ptr<MultiModelPage>> pages =
        page_allocator_->allocate(num_layers_ * 2);

    int32_t page_id = pages[0]->get_page_id();
    size_t offset = offsets_.back();
    offsets_.pop_back();

    pages = batch_map(offset, std::move(pages));

    // Initialize each page and store in avail_pages_[page_id][layer_idx]
    pages[0]->init(block_mem_size_, offset);

    // Store the vector of pages under the page_id key
    avail_pages_[page_id] = std::move(pages);
    num_free_blocks_.fetch_add(avail_pages_[page_id][0]->num_free_blocks(),
                               std::memory_order_relaxed);
  }

  // reserve block 0 for padding
  // padding_block_ = allocate();
  // CHECK_EQ(padding_block_.id(), 0) << "Padding block id should be 0";
}

std::vector<Block> MultiModelBlockManagerImpl::allocate(size_t num_blocks) {
  /*if (!has_enough_blocks(num_blocks)) {
    return {};
  }

  CHECK(num_blocks <= num_free_blocks_) << "Not enough blocks available";*/
  std::vector<Block> blocks;
  blocks.reserve(num_blocks);
  size_t remaining_need = num_blocks;

  while (remaining_need > 0) {
    std::vector<std::unique_ptr<MultiModelPage>> pages;

    if (avail_pages_.empty()) {
      pages = page_allocator_->allocate(num_layers_ * 2);

      size_t offset = offsets_.back();
      offsets_.pop_back();

      pages = batch_map(offset, std::move(pages));
      pages[0]->init(block_mem_size_, offset);

      num_free_blocks_.fetch_add(pages[0]->num_free_blocks(),
                                 std::memory_order_relaxed);
    } else {
      // Get an available page set from the map
      auto it = avail_pages_.begin();
      int32_t page_id = it->first;
      pages = std::move(it->second);
      avail_pages_.erase(it);
      LOG(INFO) << "DEBUG" << pages[0]->num_free_blocks();
    }

    // Allocate blocks from the first page (layer 0)
    size_t num_from_page =
        std::min(pages[0]->num_free_blocks(), remaining_need);
    std::vector<int32_t> alloced_blocks =
        pages[0]->alloc(num_from_page);  // pages[0]是映射层
    for (int32_t block_id : alloced_blocks) {
      blocks.emplace_back(block_id, this);
      page_id_[block_id] = pages[0]->get_page_id();
    }

    int32_t page_id = pages[0]->get_page_id();
    if (pages[0]->full()) {
      full_pages_[page_id] = std::move(pages);
    } else {
      avail_pages_[page_id] = std::move(pages);
    }

    num_free_blocks_.fetch_sub(num_from_page, std::memory_order_relaxed);
    remaining_need -= num_from_page;
  }
  num_used_blocks_.fetch_add(num_blocks, std::memory_order_relaxed);
  return blocks;
}

void MultiModelBlockManagerImpl::deallocate(const Slice<Block>& blocks) {
  /*if (options_.enable_prefix_cache()) {
    for (const auto& block : blocks) {
      // the block is not shared by other sequence
      if (block.is_valid() && block.ref_count() <= 2) {
        auto origin_num_used_blocks =
            num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
        if (origin_num_used_blocks < 0) {
          LOG(ERROR) << "num_used_blocks_==0 cannot fetch_sub for id:"
                     << block.id()
                     << ", total block size: " << num_total_blocks();
          std::unordered_set<int32_t> block_id_set;
          block_id_set.insert(block.id());
          std::string error_msg = "Block already released: ";
          for (auto& id : free_blocks_) {
            if (block_id_set.count(id) != 0) {
              error_msg.append(std::to_string(id)).append(" ");
            }
          }
          LOG(FATAL) << error_msg;
        }
      }
    }
  } else {
    num_used_blocks_.fetch_sub(blocks.size(), std::memory_order_relaxed);
  }*/
  num_used_blocks_.fetch_sub(blocks.size(), std::memory_order_relaxed);
}

bool MultiModelBlockManagerImpl::has_enough_blocks(uint32_t num_blocks) {
  return true;
  /*
    if (num_blocks <= num_free_blocks_) {
      return true;
    }

    // prefix cache is disabled, no way to evict blocks
    if (!options_.enable_prefix_cache()) {
      return false;
    }

    // try to evict some blocks from the prefix cache
    const uint32_t n_blocks_to_evict = num_blocks - num_free_blocks_;

    AUTO_COUNTER(prefix_cache_latency_seconds_evict);
    const uint32_t n_blocks_evicted = prefix_cache_->evict(n_blocks_to_evict);
    if (n_blocks_evicted < n_blocks_to_evict) {
      return false;
    }

    if (num_free_blocks_ >= num_blocks) {
      return true;
    }

    LOG(WARNING) << "Potential block leak, free blocks in allocator: "
                 << num_free_blocks_
                 << " blocks in prefix cache: " << prefix_cache_->num_blocks();
    return false;
  */
}

// caller should make sure the block_id is valid
void MultiModelBlockManagerImpl::free(int32_t block_id) {
  int32_t page_id = page_id_[block_id];
  std::vector<std::unique_ptr<MultiModelPage>> pages;
  auto it_full = full_pages_.find(page_id);
  if (it_full != full_pages_.end()) {  // in full pages
    pages = std::move(it_full->second);
    full_pages_.erase(it_full);
  } else {
    auto it_avail = avail_pages_.find(page_id);
    CHECK_EQ(it_avail != avail_pages_.end(), true)
        << "Invalid block id to free: " << block_id;
    pages = std::move(it_avail->second);
    avail_pages_.erase(it_avail);
  }
  num_free_blocks_.fetch_add(1, std::memory_order_relaxed);
  pages[0]->free(block_id);

  if (pages[0]->empty()) {
    size_t offset = pages[0]->get_offset();
    offsets_.push_back(offset);

    pages = batch_unmap(offset, std::move(pages));

    pages[0]->reset();
    page_allocator_->deallocate(std::move(pages));
  } else {
    avail_pages_[page_id] = std::move(pages);
  }
}
std::vector<std::unique_ptr<MultiModelPage>>
MultiModelBlockManagerImpl::batch_map(
    size_t offset,
    std::vector<std::unique_ptr<MultiModelPage>> pages) const {
  for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    VirPtr k_vit_ptr =
        multi_layer_k_xtensor_->get_block_vir_ptr(offset, layer_idx);
    VirPtr v_vit_ptr =
        multi_layer_v_xtensor_->get_block_vir_ptr(offset, layer_idx);
    map(k_vit_ptr, pages[layer_idx]->get_phy_handle());
    map(v_vit_ptr, pages[layer_idx + num_layers_]->get_phy_handle());
  }
  return std::move(pages);
}

std::vector<std::unique_ptr<MultiModelPage>>
MultiModelBlockManagerImpl::batch_unmap(
    size_t offset,
    std::vector<std::unique_ptr<MultiModelPage>> pages) const {
  size_t aligned_size = pages[0]->get_page_size();
  for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    VirPtr k_vit_ptr =
        multi_layer_k_xtensor_->get_block_vir_ptr(offset, layer_idx);
    VirPtr v_vit_ptr =
        multi_layer_v_xtensor_->get_block_vir_ptr(offset, layer_idx);
    unmap(k_vit_ptr, aligned_size);
    unmap(v_vit_ptr, aligned_size);
  }
  return std::move(pages);
}
// map one virtual pointer to one physical page
void MultiModelBlockManagerImpl::map(VirPtr vir_ptr,
                                     PhyMemHandle phy_handle) const {
  // TODO：refactor virtensor shape
  vmm::map(vir_ptr, phy_handle);
}

void MultiModelBlockManagerImpl::unmap(VirPtr vir_ptr,
                                       size_t aligned_size) const {
  vmm::unmap(vir_ptr, aligned_size);
}

void MultiModelBlockManagerImpl::add_multi_layer_kv_xtensor(
    torch::Device device_) {
  BlockMultiLayerXTensorPair multi_layer_kv_xtensor =
      BlockMultiLayerXTensorTransfer::get_instance().move_multi_layer_xtensor(
          device_.index(), model_idx_);
  multi_layer_k_xtensor_ = std::move(multi_layer_kv_xtensor.first);
  multi_layer_v_xtensor_ = std::move(multi_layer_kv_xtensor.second);
  num_layers_ = multi_layer_k_xtensor_->get_num_layers();
}
/*
std::vector<Block> BlockManagerImpl::allocate_shared(
    const Slice<int32_t>& tokens_ids,
    const Slice<Block>& existed_shared_blocks) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_match);

    std::vector<Block> shared_blocks =
        prefix_cache_->match(tokens_ids, existed_shared_blocks);

    const size_t prefix_length =
        shared_blocks.empty() ? 0
                              : shared_blocks.size() * shared_blocks[0].size();
    COUNTER_ADD(prefix_cache_match_length_total, prefix_length);

    // update effective block usage
    for (const auto& block : shared_blocks) {
      // the block is not shared by any sequence
      if (block.ref_count() <= 2) {
        num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return shared_blocks;
  }
  return {};
}

void BlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                             std::vector<Block>& blocks) {
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_insert);
    // Add the kv cache to the prefix cache
    prefix_cache_->insert(token_ids, blocks);
  }
}

void BlockManagerImpl::get_merged_kvcache_event(KvCacheEvent* event) const {
  auto events = prefix_cache_->get_upload_kvcache_events();
  if (events != nullptr) {
    event->removed_cache.merge(events->removed_cache);
    event->stored_cache.merge(events->stored_cache);
    event->offload_cache.merge(events->offload_cache);
    events->clear();
  }
}

// allocate a block id
Block BlockManagerImpl::allocate() {
  CHECK(num_free_blocks_ > 0) << "No more blocks available";
  size_t prev_count = num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
  const int32_t block_id = free_blocks_[prev_count - 1];
  return {block_id, this};
}


*/
}  // namespace xllm
