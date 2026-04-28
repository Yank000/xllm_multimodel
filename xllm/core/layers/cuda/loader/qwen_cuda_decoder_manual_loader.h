#pragma once

#include "base_cuda_manual_loader.h"

namespace xllm {
namespace layer {

class Qwen2DecoderLayerImpl;

class QwenCudaDecoderManualLoader : public BaseCudaManualLoader {
 public:
  explicit QwenCudaDecoderManualLoader(const ModelContext& context);

  void bind(Qwen2DecoderLayerImpl& layer);
};

}  // namespace layer
}  // namespace xllm
