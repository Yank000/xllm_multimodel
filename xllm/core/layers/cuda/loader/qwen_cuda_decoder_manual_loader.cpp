#include "qwen_cuda_decoder_manual_loader.h"

#include "core/layers/qwen2_decoder_layer.h"

namespace xllm {
namespace layer {

QwenCudaDecoderManualLoader::QwenCudaDecoderManualLoader(
    const ModelContext& context)
    : BaseCudaManualLoader(context) {}

void QwenCudaDecoderManualLoader::bind(Qwen2DecoderLayerImpl& layer) {
  auto& attention = layer.attention_for_cuda_loader();
  auto& mlp = layer.mlp_for_cuda_loader();

  bind_weight("input_layernorm.weight",
              &layer.input_norm_for_cuda_loader()->mutable_weight_tensor());
  bind_weight("self_attn.qkv_proj.weight",
              &attention->qkv_proj_for_cuda_loader()->mutable_weight_tensor());
  bind_weight("self_attn.o_proj.weight",
              &attention->o_proj_for_cuda_loader()->mutable_weight_tensor());
  if (attention->is_qwen3_style_for_cuda_loader()) {
    bind_weight("self_attn.q_norm.weight",
                &attention->q_norm_for_cuda_loader()->mutable_weight_tensor());
    bind_weight("self_attn.k_norm.weight",
                &attention->k_norm_for_cuda_loader()->mutable_weight_tensor());
  }
  bind_weight("post_attention_layernorm.weight",
              &layer.post_norm_for_cuda_loader()->mutable_weight_tensor());
  bind_weight("mlp.gate_up_proj.weight",
              &mlp->gate_up_proj_for_cuda_loader()->mutable_weight_tensor());
  bind_weight("mlp.down_proj.weight",
              &mlp->down_proj_for_cuda_loader()->mutable_weight_tensor());
}

}  // namespace layer
}  // namespace xllm
