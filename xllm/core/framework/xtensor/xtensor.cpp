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

#include "xtensor.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/util/tensor_helper.h"
#include "phy_page_pool.h"
#include "platform/vmm_api.h"

namespace xllm {

// Align size up to page_size granularity
static inline size_t align_up(size_t size, size_t page_size) {
  return ((size + page_size - 1) / page_size) * page_size;
}

static inline VirPtr alloc_virtual_mem(size_t size) {
  size_t page_size = FLAGS_phy_page_granularity_size;
  CHECK(size % page_size == 0)
      << "alloc size not aligned: " << size;  // Ensure alignment.

  VirPtr vaddr;
  vmm::create_vir_ptr(vaddr, size);
  return vaddr;
}

XTensor::XTensor(size_t size,
                 torch::Dtype dtype,
                 torch::Device dev,
                 PhyPage* zero_page)
    : vaddr_(0),
      size_(0),
      page_size_(FLAGS_phy_page_granularity_size),
      dtype_(dtype),
      dev_(dev),
      zero_page_(zero_page) {
  // Align size to page_size_
  size_ = align_up(size, page_size_);
  vaddr_ = alloc_virtual_mem(size_);
  init_with_zero_();
}

XTensor::~XTensor() {
  // Collect all physical pages to return in batch
  worker_running_ = false;
  std::vector<std::unique_ptr<PhyPage>> pages_to_return;
  pages_to_return.reserve(mapping_.size());
  for (auto& [page_id, page] : mapping_) {
    pages_to_return.push_back(std::move(page));
  }
  mapping_.clear();

  // Return all pages to pool in one lock
  if (!pages_to_return.empty()) {
    PhyPagePool::get_instance().batch_put(pages_to_return);
  }
  // zero_page_ is not owned, don't delete it

  if (vaddr_) {
    // Unmap all physical pages first
    for (size_t offset = 0; offset < size_; offset += page_size_) {
      VirPtr addr = reinterpret_cast<VirPtr>(
          reinterpret_cast<uintptr_t>(vaddr_) + offset);
      vmm::unmap(addr, page_size_);
    }
    // Release virtual memory
    vmm::release_vir_ptr(vaddr_, size_);
  }
}

bool XTensor::map(offset_t offset) {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;

  page_id_t page_id = offset / page_size_;

  // Check if already mapped (idempotent: return true if already mapped)
  if (mapping_.find(page_id) != mapping_.end()) {
    return true;
  }

  // Get a physical page from pool
  auto phy_pages = PhyPagePool::get_instance().batch_get(1);
  if (phy_pages.empty()) {
    LOG(ERROR) << "Failed to get physical page from pool";
    return false;
  }

  // Map the physical page
  VirPtr vaddr =
      reinterpret_cast<VirPtr>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  vmm::unmap(vaddr, page_size_);

  PhyMemHandle phy_handle = phy_pages[0]->get_phy_handle();
  vmm::map(vaddr, phy_handle);

  mapping_[page_id] = std::move(phy_pages[0]);
  return true;
}

bool XTensor::unmap(offset_t offset) {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;

  page_id_t page_id = offset / page_size_;

  auto it = mapping_.find(page_id);
  if (it == mapping_.end()) {
    // Already unmapped (idempotent: return true)
    return true;
  }

  VirPtr vaddr =
      reinterpret_cast<VirPtr>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  vmm::unmap(vaddr, page_size_);

  // Map the zero page instead to ensure memory integrity
  map_phy_page_(zero_page_, offset);

  // Return the physical page to pool
  std::vector<std::unique_ptr<PhyPage>> pages_to_return;
  pages_to_return.push_back(std::move(it->second));
  mapping_.erase(it);
  PhyPagePool::get_instance().batch_put(pages_to_return);

  return true;
}

void XTensor::worker() {
  while (worker_running_) {
    // 检查是否需要进行map操作
    if (alloc_offset_ < target_ * page_size_) {  // map
      if (!map(alloc_offset_)) {
        LOG(ERROR) << "Failed to map page at offset " << alloc_offset_;
      } else {
        alloc_offset_ += page_size_;
        // empty_size_mb += page_size_;
      }
    } else if (alloc_offset_ > target_ * page_size_) {  // unmap
      size_t offset = alloc_offset_ - page_size_;
      if (!unmap(offset)) {
        LOG(ERROR) << "Failed to unmap page at offset " << offset;
      } else {
        // empty_size_mb -= page_size_;
        alloc_offset_ -= page_size_;
      }
    }

    // 通过条件变量通知等待中的allocate_activation
    {
      std::lock_guard<std::mutex> lock(mtx_);
      cond_.notify_all();
    }
  }
}

void XTensor::start_worker_thread() {
  std::thread worker_thread(&XTensor::worker, this);
  worker_thread.detach();  // 分离线程，确保它不会阻塞主线程
}

bool XTensor::map_all() {
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    if (!map(offset)) {
      LOG(ERROR) << "Failed to map page at offset " << offset;
      return false;
    }
  }
  return true;
}

bool XTensor::unmap_all() {
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    page_id_t page_id = offset / page_size_;
    // Only unmap if the page is mapped
    if (mapping_.find(page_id) != mapping_.end()) {
      if (!unmap(offset)) {
        LOG(ERROR) << "Failed to unmap page at offset " << offset;
        return false;
      }
    }
  }
  return true;
}

bool XTensor::map_phy_page_(PhyPage* page, offset_t offset) {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;
  CHECK(page) << "Page is null";

  VirPtr vaddr =
      reinterpret_cast<VirPtr>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  return true;
}

bool XTensor::init_with_zero_() {
  CHECK(reinterpret_cast<uintptr_t>(vaddr_) % page_size_ == 0)
      << "vaddr not aligned to page size";
  CHECK(size_ % page_size_ == 0) << "size not aligned to page size";

  bool succ = true;

  // Initialize all pages with zero page
  for (size_t offset = 0; offset < size_; offset += page_size_) {
    if (!map_phy_page_(zero_page_, offset)) {
      succ = false;
      break;
    }
  }
  return succ;
}

bool XTensor::allocate_activation(void*& ptr, size_t size) {
  size_t offset = best_fit(size);

  if (offset + size > size_) {
    LOG(ERROR) << "XTensor::allocate failed: requested " << size
               << " bytes at offset " << alloc_offset_ << ", but only "
               << (size_ - alloc_offset_) << " bytes available"
               << " (total size: " << size_ << ")";
    return false;
  }

  if (offset + size > target_ * page_size_) {
    target_ = (offset + size + page_size_ - 1) / page_size_;
    if (!worker_running_) {
      worker_running_ = true;
      start_worker_thread();
    }
  }
  while (offset + size > alloc_offset_) {
    std::unique_lock<std::mutex> lock(mtx_);
    cond_.wait(lock);
  }
  // Check if already mapped
  /*int64_t instant_map_size = offset + size - alloc_offset_;

  if (instant_map_size > 0) {
    size_t num_pages = (instant_map_size + page_size_ - 1) / page_size_;
    for (int i = 0; i < num_pages; i++) {
      if(!map(alloc_offset_)) {
        LOG(ERROR) << "XTensor::allocate_activation: map_pages failed
  insufficient physical pages"
                << "for offset=" << alloc_offset_;
        return false;
      }
      alloc_offset_ += page_size_;
    }
  }*/
  // empty_size_mb -= size;
  // LOG(INFO) << empty_size_mb/2048/1024 << "MB left";
  ptr = reinterpret_cast<VirPtr>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  // LOG(INFO) << "[allocate]: pageSize=" << size / page_size_ << ", pageId=" <<
  // offset/page_size_;

  // Async map reserved pages
  // 模型初始化的时候就会调用该函数，所以当执行请求的时候已经存在预留池
  /*
  if (map_weight_tensorsync_map_pages > 0) {
    std::thread t([this, async_map_pages]() {
      while (alloc_offset_ < async_map_pages * page_size_) {
          LOG(INFO) << "XTensor::deallocate: async unmap page at offset " <<
  alloc_offset_/page_size_; map(alloc_offset_); alloc_offset_ += page_size_;
      }
    });
    t.detach();
  }*/

  // LOG(INFO) << "XTensor::allocate_activation: size=" <<
  // alloc_offset_/page_size_*2;

  return true;
}

bool XTensor::deallocate_activation(void*& ptr, size_t size) {
  // TODO: 实现收缩内存的逻辑
  size_t offset =
      reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(vaddr_);
  // 查找目标内存块
  auto it = allocated_blocks.lower_bound(memory_block{offset, 0});
  if (it == allocated_blocks.end() || it->offset != offset ||
      it->size != size) {
    // LOG(ERROR) << "XTensor::deallocate failed: no matching block found for
    // offset " << offset << " and size " << size;
    return false;  // 没有找到对应的内存块
  }

  // 释放目标内存块
  memory_block target_block = *it;
  allocated_blocks.erase(it);
  // empty_size_mb += size;

  return true;
}
// 386mb/947
size_t XTensor::best_fit(size_t request_size) {
  size_t best_fit_offset = -1;
  size_t best_fit_gap = size_ + 1;  // 初始值设为不可能的最大值

  // 遍历已分配的内存块，查找合适的空闲区域
  for (auto it = allocated_blocks.begin(); it != allocated_blocks.end(); ++it) {
    size_t current_offset = it->offset;
    size_t current_size = it->size;

    // 检查分配块之间的空隙
    if (it != allocated_blocks.begin()) {
      auto prev_it = std::prev(it);
      size_t gap = current_offset - (prev_it->offset + prev_it->size);
      if (gap >= request_size && gap < best_fit_gap) {
        best_fit_gap = gap;
        best_fit_offset = prev_it->offset + prev_it->size;
      }
    }
  }

  // 检查最后一块内存之后的空隙
  if (!allocated_blocks.empty()) {
    auto last_block = *allocated_blocks.rbegin();
    size_t gap = size_ - (last_block.offset + last_block.size);
    if (gap >= request_size && gap < best_fit_gap) {
      best_fit_gap = gap;
      best_fit_offset = last_block.offset + last_block.size;
    }
  } else {
    // 如果没有任何内存分配，直接从0开始
    best_fit_offset = 0;
  }

  // 如果找到合适的空闲空间
  if (best_fit_offset != -1) {
    allocated_blocks.insert(memory_block{best_fit_offset, request_size});
    return best_fit_offset;
  }
  // 如果没有合适的空闲空间，返回最后一个分配内存块的结束位置
  if (!allocated_blocks.empty()) {
    auto last_block = *allocated_blocks.rbegin();
    best_fit_offset = last_block.offset + last_block.size;
    allocated_blocks.insert(memory_block{best_fit_offset, request_size});
    return best_fit_offset;
  }
  // 如果没有内存分配，返回 0
  return 0;
}

bool XTensor::allocate(void*& ptr, size_t size) {
  // Check if there's enough space
  if (alloc_offset_ + size > size_) {
    LOG(ERROR) << "XTensor::allocate failed: requested " << size
               << " bytes at offset " << alloc_offset_ << ", but only "
               << (size_ - alloc_offset_) << " bytes available"
               << " (total size: " << size_ << ")";
    return false;
  }

  ptr =
      reinterpret_cast<void*>(reinterpret_cast<char*>(vaddr_) + alloc_offset_);
  // Update allocation offset
  alloc_offset_ += size;

  VLOG(2) << "XTensor::allocate: size=" << size
          << ", new_alloc_offset=" << alloc_offset_;

  return true;
}

torch::Tensor XTensor::to_torch_tensor() const {
  auto num_elems = static_cast<int64_t>(size_ / torch::elementSize(dtype_));
  return to_torch_tensor(0, {num_elems});
}

page_id_t XTensor::get_phy_page_id(offset_t offset) const {
  CHECK(offset % page_size_ == 0)
      << "Offset not aligned to page size: " << offset;

  page_id_t local_page_id = offset / page_size_;
  auto it = mapping_.find(local_page_id);
  if (it == mapping_.end()) {
    // Not mapped, return -1
    return -1;
  }
  return it->second->page_id();
}

torch::Tensor XTensor::to_torch_tensor(size_t offset,
                                       const std::vector<int64_t>& dims) const {
  uintptr_t addr = reinterpret_cast<uintptr_t>(vaddr_) + offset;
  auto dtype = dtype_;

#if defined(USE_NPU)
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  auto tensor = torch::empty({0}, option);
  auto address = reinterpret_cast<void*>(addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
  storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
  storage.set_data_ptr(std::move(c10_data_ptr));

  tensor.set_(storage, 0, dims);

  return tensor;
#else
  // For non-NPU devices, use torch::from_blob
  auto options =
      torch::TensorOptions().dtype(dtype).device(dev_).requires_grad(false);
  return torch::from_blob(reinterpret_cast<void*>(addr), dims, options);
#endif
}

}  // namespace xllm
