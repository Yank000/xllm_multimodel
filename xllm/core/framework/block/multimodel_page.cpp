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

#include "multimodel_page.h"

namespace xllm {
MultiModelPage::MultiModelPage(torch::Device device,
                               int32_t page_id,
                               int32_t page_size)
    : device_(device),
      page_size_(page_size),
      page_id_(page_id),
      start_block_(0),
      end_block_(0),
      num_kv_blocks_(0) {
  int32_t device_id = device_.index();
  // create a physical memory handle for the device
  vmm::create_phy_mem_handle(phy_handle_, device_id, page_size);
}

MultiModelPage::~MultiModelPage() { vmm::release_phy_mem_handle(phy_handle_); }

void MultiModelPage::_require_init() {
  // Raise Error if the page has not been initialised.
  CHECK_NE(end_block_, 0) << "Page not initialised";
  CHECK_NE(num_kv_blocks_, 0) << "Page not initialised";
}

void MultiModelPage::init(size_t block_mem_size) {
  set_block_range(block_mem_size);

  num_kv_blocks_ = end_block_ - start_block_;
  for (size_t i = start_block_; i < end_block_; ++i) {
    free_list_.push_back(i);
  }
}

void MultiModelPage::reset() {
  free_list_.clear();
  start_block_ = 0;
  end_block_ = 0;
  num_kv_blocks_ = 0;
}

void MultiModelPage::set_block_range(size_t block_mem_size) {
  /*
  Get the block range of a page.
        The page contains [start_block, end_block), which handles the case where
        page_size is not divisible by block_mem_size.
        For example, if page_size = 16 and block_mem_size = 6, the page 0
        contains [0, 2) blocks, and the page 1 contains [3, 5) blocks.
        Pages:  |      0-16       |        16-32        |
                | 0-6 | 6-12 | 12-18 | 18-24 | 24-30 | 30-32 |
        Blocks: |  0  |  1   |2<skip>|   3   |   4   |5<skip>|
  */
  start_block_ = (page_id_ * page_size_ + block_mem_size - 1) / block_mem_size;
  end_block_ = ((page_id_ + 1) * page_size_) / block_mem_size;
}

size_t MultiModelPage::num_free_blocks() {
  _require_init();
  return free_list_.size();
}

std::vector<int32_t> MultiModelPage::alloc(size_t num_blocks) {
  _require_init();
  CHECK_EQ(full(), false) << "Page " << page_id_ << " is already full";
  std::vector<int32_t> blocks;
  for (size_t i = 0; i < num_blocks; ++i) {
    blocks.push_back(free_list_.back());
    free_list_.pop_back();
  }
  return blocks;
}

void MultiModelPage::free(size_t block_id) {
  _require_init();
  CHECK_EQ(_has_block(block_id), true)
      << "Block id " << block_id << " out of range for page " << page_id_;
  CHECK_EQ(_in_free_list(block_id), false)
      << "Block id " << block_id << " already freed in page " << page_id_;
  free_list_.push_back(block_id);
}

bool MultiModelPage::_has_block(size_t block_id) {
  return block_id >= start_block_ && block_id < end_block_;
}

bool MultiModelPage::_in_free_list(size_t block_id) {
  return std::find(free_list_.begin(), free_list_.end(), block_id) !=
         free_list_.end();
}

bool MultiModelPage::full() {
  _require_init();
  return free_list_.empty();
}

bool MultiModelPage::empty() {
  _require_init();
  return free_list_.size() == num_kv_blocks_;
}

}  // namespace xllm