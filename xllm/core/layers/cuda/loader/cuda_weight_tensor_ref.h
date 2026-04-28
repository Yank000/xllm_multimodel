#pragma once

#include <torch/torch.h>

#include <string>

namespace xllm {
namespace layer {

struct CudaWeightTensorRef {
  std::string name;
  torch::Tensor* tensor = nullptr;
  bool required = true;
};

}  // namespace layer
}  // namespace xllm
