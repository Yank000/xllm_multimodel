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
// TODO: group pages by layer, fix slot size issue
#include "multimodel_page_pool.h"

#include "common/global_flags.h"
namespace xllm {

MultiModelPagePool::MultiModelPagePool(
    const MultiModelPagePool::Options& options,
    const torch::Device& device)
    : options_(options), device_(device) {
  CHECK_GT(options_.num_total_pages(), 0) << "No pages to allocate";
  // 目前为静态分配，处理为每个模型物理页均分

  page_size_ =
      2 * 1024 *
      1024;  // TODO:decide the proper page_size_, such as LCM of 1,2,and 2MB

  LOG(INFO) << "MultiModelPagePool: total pages " << options_.num_total_pages()
            << ", num models " << options_.num_models()
            << ", Granularity size for physical page: " << page_size_ << "MB.";

  num_free_phy_pages_.fetch_add(options_.num_total_pages());

  free_phy_pages_.reserve(options_.num_total_pages());
  for (int64_t i = 0; i < options_.num_total_pages(); ++i) {
    free_phy_pages_.push_back(
        std::make_unique<MultiModelPage>(device_, i, page_size_));
  }
}

std::vector<std::unique_ptr<MultiModelPage>> MultiModelPagePool::allocate(
    int32_t num_needed) {
  CHECK_GT(num_free_phy_pages_, 0)
      << "Not enough physical pages available";  // 第一阶段，静态比例

  std::vector<std::unique_ptr<MultiModelPage>> pages;
  pages.reserve(num_needed);

  uint32_t page_id =
      num_free_phy_pages_.fetch_sub(num_needed, std::memory_order_relaxed) -
      num_needed;
  LOG(INFO) << "DEBUG" << page_id << " " << page_id / 2;
  // Allocate one page per layer
  for (int64_t i = 0; i < num_needed; ++i) {
    pages.push_back(std::move(free_phy_pages_[page_id + i]));
  }

  return pages;
}

// caller should make sure the page_id is valid
void MultiModelPagePool::deallocate(
    std::vector<std::unique_ptr<MultiModelPage>> pages) {
  int64_t page_id =
      num_free_phy_pages_.fetch_add(pages.size(), std::memory_order_relaxed);
  //  Deallocate pages back to their respective layers
  for (size_t layer_idx = 0; layer_idx < pages.size(); ++layer_idx) {
    free_phy_pages_[page_id + layer_idx] = std::move(pages[layer_idx]);
  }
}
}  // namespace xllm
