#include "base_cuda_manual_loader.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstring>

#include "core/common/global_flags.h"
#include "core/framework/xtensor/xtensor_allocator.h"

namespace xllm {
namespace layer {
namespace {

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

void check_cuda(cudaError_t err, const char* what) {
  CHECK_EQ(err, cudaSuccess) << what << " failed: " << cudaGetErrorString(err);
}

cudaStream_t current_stream(const torch::Device& device) {
  CHECK(device.is_cuda()) << "CUDA manual loader requires cuda device";
  c10::cuda::CUDAGuard guard(device);
  return at::cuda::getCurrentCUDAStream();
}

}  // namespace

BaseCudaManualLoader::BaseCudaManualLoader(const ModelContext& context)
    : model_id_(context.get_model_id()),
      device_(context.get_tensor_options().device()) {}

BaseCudaManualLoader::~BaseCudaManualLoader() {
  release_host_storage();
  release_device_storage();
}

void BaseCudaManualLoader::bind_weight(const std::string& name,
                                       torch::Tensor* tensor,
                                       bool required) {
  refs_.push_back(CudaWeightTensorRef{name, tensor, required});
}

void BaseCudaManualLoader::initialize_from_bound_tensors() {
  init_weight_slices();
  if (storage_size_ == 0) {
    return;
  }
  copy_weights_to_pinned_host();
  allocate_device_storage();
  copy_weights_to_device_async();
  check_cuda(cudaStreamSynchronize(current_stream(device_)),
             "cudaStreamSynchronize");
  rebuild_tensor_views();
  weight_pages_on_device_ = true;
}

void BaseCudaManualLoader::init_weight_slices() {
  weight_slices_.resize(refs_.size());
  size_t offset = 0;
  for (size_t i = 0; i < refs_.size(); ++i) {
    weight_slices_[i] = {};
    const auto& ref = refs_[i];
    CHECK(ref.tensor != nullptr) << "null tensor ref: " << ref.name;
    const torch::Tensor& tensor = *ref.tensor;
    if (!tensor.defined() || tensor.numel() == 0) {
      CHECK(!ref.required) << "required tensor is not defined: " << ref.name;
      continue;
    }
    CHECK(tensor.is_contiguous()) << "CUDA manual loader requires contiguous "
                                  << "weight tensor: " << ref.name;
    offset = align_up(offset, kHostAlignment);
    weight_slices_[i].offset = offset;
    weight_slices_[i].bytes = tensor.nbytes();
    weight_slices_[i].sizes = tensor.sizes().vec();
    weight_slices_[i].dtype = tensor.scalar_type();
    offset += weight_slices_[i].bytes;
  }
  const size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = align_up(offset, max_alignment);
}

void BaseCudaManualLoader::copy_weights_to_pinned_host() {
  c10::cuda::CUDAGuard guard(device_);
  CHECK_GT(storage_size_, 0U) << "storage size must be greater than 0";
  check_cuda(cudaMallocHost(&host_pinned_storage_, storage_size_),
             "cudaMallocHost");

  for (size_t i = 0; i < refs_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (slice.bytes == 0) {
      continue;
    }
    const auto& ref = refs_[i];
    auto tensor = ref.tensor->contiguous();
    void* dst = static_cast<char*>(host_pinned_storage_) + slice.offset;
    if (tensor.is_cuda()) {
      check_cuda(
          cudaMemcpy(
              dst, tensor.data_ptr(), slice.bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
    } else {
      std::memcpy(dst, tensor.data_ptr(), slice.bytes);
    }
  }
}

void BaseCudaManualLoader::allocate_device_storage() {
  c10::cuda::CUDAGuard guard(device_);
  if (device_storage_ != nullptr) {
    return;
  }
  if (FLAGS_enable_xtensor &&
      XTensorAllocator::get_instance().is_initialized()) {
    auto& allocator = XTensorAllocator::get_instance();
    bool ok =
        allocator.allocate_weight(model_id_, device_storage_, storage_size_);
    CHECK(ok) << "Failed to allocate XTensor weight storage size="
              << storage_size_;
    owns_device_storage_ = false;
    return;
  }
  check_cuda(cudaMalloc(&device_storage_, storage_size_), "cudaMalloc");
  owns_device_storage_ = true;
}

void BaseCudaManualLoader::copy_weights_to_device_async() {
  c10::cuda::CUDAGuard guard(device_);
  CHECK(device_storage_ != nullptr) << "device storage is not allocated";
  CHECK(host_pinned_storage_ != nullptr) << "host pinned storage is null";
  check_cuda(cudaMemcpyAsync(device_storage_,
                             host_pinned_storage_,
                             storage_size_,
                             cudaMemcpyHostToDevice,
                             at::cuda::getCurrentCUDAStream()),
             "cudaMemcpyAsync H2D");
}

void BaseCudaManualLoader::rebuild_tensor_views() {
  CHECK(device_storage_ != nullptr) << "device storage is not allocated";
  for (size_t i = 0; i < refs_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (slice.bytes == 0) {
      continue;
    }
    void* base = static_cast<char*>(device_storage_) + slice.offset;
    auto options = torch::TensorOptions().dtype(slice.dtype).device(device_);
    torch::Tensor view =
        torch::from_blob(base, slice.sizes, [](void*) {}, options);
    *refs_[i].tensor = view;
  }
}

int64_t BaseCudaManualLoader::release_weight_pages_for_this_layer() {
  if (device_storage_ == nullptr || storage_size_ == 0 ||
      !weight_pages_on_device_) {
    return 0;
  }
  if (FLAGS_enable_xtensor &&
      XTensorAllocator::get_instance().is_initialized()) {
    auto& allocator = XTensorAllocator::get_instance();
    size_t n = allocator.unmap_weight_region(
        model_id_, device_storage_, storage_size_);
    size_t reclaimed =
        allocator.reclaim_mapped_zero_ref_weight_pages(model_id_);
    if (n > 0 || reclaimed > 0) {
      weight_pages_on_device_ = false;
    }
    return static_cast<int64_t>(n + reclaimed);
  }
  release_device_storage();
  weight_pages_on_device_ = false;
  return 0;
}

int64_t BaseCudaManualLoader::ensure_weight_pages_mapped_then_copy_from_host() {
  if (storage_size_ == 0) {
    return 0;
  }
  if (device_storage_ == nullptr) {
    allocate_device_storage();
  }
  int64_t pages_mapped = 0;
  if (FLAGS_enable_xtensor &&
      XTensorAllocator::get_instance().is_initialized()) {
    auto& allocator = XTensorAllocator::get_instance();
    pages_mapped = allocator.ensure_weight_pages_mapped_region(
        model_id_, device_storage_, storage_size_);
    if (pages_mapped < 0) {
      return -1;
    }
  }
  copy_weights_to_device_async();
  check_cuda(cudaStreamSynchronize(current_stream(device_)),
             "cudaStreamSynchronize");
  rebuild_tensor_views();
  weight_pages_on_device_ = true;
  return pages_mapped;
}

void BaseCudaManualLoader::release_host_storage() {
  c10::cuda::CUDAGuard guard(device_);
  if (host_pinned_storage_ != nullptr) {
    check_cuda(cudaFreeHost(host_pinned_storage_), "cudaFreeHost");
    host_pinned_storage_ = nullptr;
  }
}

void BaseCudaManualLoader::release_device_storage() {
  c10::cuda::CUDAGuard guard(device_);
  if (device_storage_ != nullptr && owns_device_storage_) {
    check_cuda(cudaFree(device_storage_), "cudaFree");
  }
  device_storage_ = nullptr;
  owns_device_storage_ = false;
}

}  // namespace layer
}  // namespace xllm
