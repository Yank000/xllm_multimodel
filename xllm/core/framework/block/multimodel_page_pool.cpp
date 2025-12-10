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
// TODO:right now,you have xtensors vector;你得先分配所有的worker，再启动该东西
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
}

void MultiModelPagePool::init() {
  size_t num_free_phy_pages_per_model_ =
      options_.num_total_pages() / options_.num_models();
  add_multi_layer_kv_xtensors(device_);
  num_layers_.reserve(options_.num_models());
  for (int64_t i = 0; i < options_.num_models(); ++i) {
    num_layers_.push_back(multi_layer_kv_xtensors_.first[i]->get_num_layers());
  }
  // Initialize vector with free page counts (construct with size to avoid
  // resize)
  num_free_phy_pages_ =
      std::vector<std::atomic<uint32_t>>(options_.num_models());

  free_phy_pages_.resize(options_.num_models());
  for (int i = 0; i < options_.num_models(); ++i) {
    free_phy_pages_[i].resize(num_layers_[i]);
    uint32_t num_free_pages =
        num_free_phy_pages_per_model_ / num_layers_[i] / 2 * 2;
    num_free_phy_pages_[i].store(num_free_pages);
    for (int j = 0; j < num_layers_[i]; ++j) {
      free_phy_pages_[i][j].resize(num_free_pages);
    }
  }
  for (int64_t i = 0; i < options_.num_models(); ++i) {
    LOG(INFO) << "MultiModelPagePool: model " << i << ", pages "
              << num_free_phy_pages_per_model_ / num_layers_[i] / 2 * 2
              << " allocating.";
    for (int64_t j = 0; j < num_layers_[i]; ++j) {
      for (int64_t k = 0; k < free_phy_pages_[i][j].size();
           ++k) {  // 静态分配两块KV Tensor，所以page_id不唯一
        free_phy_pages_[i][j][k] = std::make_unique<MultiModelPage>(
            device_, k, page_size_);  // 2MB default
      }
    }
  }
}

int32_t MultiModelPagePool::get_page_id(int32_t block_id,
                                        size_t block_mem_size) {
  return block_id * block_mem_size / page_size_;
}

void MultiModelPagePool::add_multi_layer_kv_xtensors(torch::Device device_) {
  multi_layer_kv_xtensors_ =
      BlockMultiLayerXTensorTransfer::get_instance().move_multi_layer_xtensors(
          device_.index());
}

std::vector<std::unique_ptr<MultiModelPage>> MultiModelPagePool::allocate(
    int32_t model_idx) {
  CHECK_GT(num_free_phy_pages_[model_idx], 0)
      << "Not enough physical pages available";  // 第一阶段，静态比例

  std::vector<std::unique_ptr<MultiModelPage>> pages;
  pages.reserve(num_layers_[model_idx]);
  // TODO：add map logic

  auto& multi_layer_k_xtensor = multi_layer_kv_xtensors_.first[model_idx];
  auto& multi_layer_v_xtensor = multi_layer_kv_xtensors_.second[model_idx];

  uint32_t page_id =
      num_free_phy_pages_[model_idx].fetch_sub(2, std::memory_order_relaxed) -
      2;
  LOG(INFO) << "DEBUG" << page_id << " " << page_id / 2;
  // Allocate one page per layer
  for (int64_t layer_idx = 0; layer_idx < num_layers_[model_idx]; layer_idx++) {
    std::unique_ptr<MultiModelPage> page =
        std::move(free_phy_pages_[model_idx][layer_idx][page_id]);

    size_t offset = page_size_ * (page_id / 2);
    VirPtr k_vit_ptr =
        multi_layer_k_xtensor->get_block_vir_ptr(offset, layer_idx);
    VirPtr v_vit_ptr =
        multi_layer_v_xtensor->get_block_vir_ptr(offset, layer_idx);

    map(k_vit_ptr, page->get_phy_handle());
    map(v_vit_ptr,
        free_phy_pages_[model_idx][layer_idx][page_id + 1]->get_phy_handle());

    pages.push_back(std::move(page));
  }

  return pages;
}

// caller should make sure the page_id is valid
void MultiModelPagePool::deallocate(
    std::vector<std::unique_ptr<MultiModelPage>> pages,
    int32_t model_idx) {
  // int64_t page_id = num_free_phy_pages_[model_idx].fetch_add(2,
  // std::memory_order_relaxed);
  //  Deallocate pages back to their respective layers
  for (size_t layer_idx = 0; layer_idx < pages.size(); ++layer_idx) {
    free_phy_pages_[model_idx][layer_idx][page_id] =
        std::move(pages[layer_idx]);
  }
  // TODO: add unmap logic
}

// map one virtual pointer to one physical page
void MultiModelPagePool::map(VirPtr vir_ptr, PhyMemHandle phy_handle) const {
  // TODO：refactor virtensor shape
  vmm::map(vir_ptr, phy_handle);
}
/*
void PhyPagePool::unmap(VirPtr vir_ptr, PhyMemHandle phy_handle) const {
  vmm::map(vir_ptr, phy_handle);
}

void PhyPagePool::unmap(VirPtr vir_ptr,
                      uint32_t page_id,
                      int64_t layer_idx) const {
  PhyMemHandle phy_handle =
      free_phy_pages_[layer_idx][page_id]->get_phy_handle();
  map(vir_ptr, phy_handle);
}
*/
/*
void PhyPagePool::batch_map(VirPtr vir_ptr,
                            std::vector<uint32_t>& page_ids,
                            uint32_t num_new_pages,
                            int64_t layer_idx) const {
  size_t num_pages = page_ids.size();

  size_t ptr_offset =
      (num_pages - num_new_pages) * FLAGS_phy_page_granularity_size;

  VirPtr temp_vir_ptr = reinterpret_cast<VirPtr>(vir_ptr + ptr_offset);

  for (size_t j = num_new_pages; j > 0; --j) {
    uint32_t page_id = page_ids[num_pages - j];
    map(temp_vir_ptr, page_id, layer_idx);
    temp_vir_ptr = reinterpret_cast<VirPtr>(temp_vir_ptr +
                                            FLAGS_phy_page_granularity_size);
  }
}
  */
}  // namespace xllm
