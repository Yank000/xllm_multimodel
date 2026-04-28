#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/model_context.h"
#include "cuda_weight_tensor_ref.h"

namespace xllm {
namespace layer {

class BaseCudaManualLoader {
 public:
  explicit BaseCudaManualLoader(const ModelContext& context);
  virtual ~BaseCudaManualLoader();

  void bind_weight(const std::string& name,
                   torch::Tensor* tensor,
                   bool required = true);

  void initialize_from_bound_tensors();
  int64_t release_weight_pages_for_this_layer();
  int64_t ensure_weight_pages_mapped_then_copy_from_host();
  bool are_weight_pages_on_device() const { return weight_pages_on_device_; }

 protected:
  struct WeightSlice {
    size_t offset = 0;
    size_t bytes = 0;
    std::vector<int64_t> sizes;
    torch::ScalarType dtype = torch::kFloat16;
  };

  void init_weight_slices();
  void copy_weights_to_pinned_host();
  void allocate_device_storage();
  void copy_weights_to_device_async();
  void rebuild_tensor_views();
  void release_host_storage();
  void release_device_storage();

  std::string model_id_;
  torch::Device device_;
  std::vector<CudaWeightTensorRef> refs_;
  std::vector<WeightSlice> weight_slices_;
  void* host_pinned_storage_ = nullptr;
  void* device_storage_ = nullptr;
  size_t storage_size_ = 0;
  bool weight_pages_on_device_ = true;
  bool owns_device_storage_ = false;

  static constexpr size_t kDeviceAlignment = 64;
  static constexpr size_t kHostAlignment = 64;
};

}  // namespace layer
}  // namespace xllm
