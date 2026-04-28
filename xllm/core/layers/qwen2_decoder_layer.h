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

#include <torch/torch.h>

#include <memory>
#include <optional>
#include <tuple>

#include "common/dense_mlp.h"
#include "common/qwen2_attention.h"
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA)
#include "cuda/loader/qwen_cuda_decoder_manual_loader.h"
#endif

namespace xllm {
namespace layer {

class Qwen2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen2DecoderLayerImpl(const ModelContext& context,
                                 int32_t layer_id = -1);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

#if defined(USE_CUDA)
  int64_t offload_weights();
  int64_t load_weights_from_pinned();
  bool are_weight_pages_on_device() const;

  Qwen2Attention& attention_for_cuda_loader() { return attention_; }
  DenseMLP& mlp_for_cuda_loader() { return mlp_; }
  RMSNorm& input_norm_for_cuda_loader() { return input_norm_; }
  RMSNorm& post_norm_for_cuda_loader() { return post_norm_; }
#endif

 private:
  Qwen2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

#if defined(USE_CUDA)
  std::unique_ptr<QwenCudaDecoderManualLoader> cuda_loader_;
#endif

  ParallelArgs parallel_args_;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> apply_norm(
      RMSNorm& norm,
      torch::Tensor& input,
      std::optional<torch::Tensor>& residual,
      const std::optional<torch::Tensor>& fp8_scale);
};
TORCH_MODULE(Qwen2DecoderLayer);

}  // namespace layer
}  // namespace xllm
