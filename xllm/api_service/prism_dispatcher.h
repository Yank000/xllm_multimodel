/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <deque>

namespace xllm {

class PrismDispatcher {
 public:
  static PrismDispatcher& instance();

  void enqueue_request(const std::string& model_name,
                       int32_t slo_ms,
                       size_t prompt_len,
                       std::function<void()> dispatch_fn);
  void mark_request_finished(const std::string& model_name);

  static double calculate_priority_value(double arrival_seconds,
                                         int32_t slo_ms,
                                         size_t prompt_len);

  void reset_for_test();
  size_t queued_size_for_test() const;
  size_t inflight_for_test(const std::string& model_name) const;

 private:
  struct RequestWrapper {
    std::string model_name;
    double priority_value = 0.0;
    uint64_t sequence = 0;
    std::function<void()> dispatch_fn;
  };

  std::vector<std::function<void()>> admit_requests_locked();

 private:
  mutable std::mutex mutex_;
  std::deque<std::shared_ptr<RequestWrapper>> queue_;
  std::unordered_map<std::string, size_t> inflight_by_model_;
  uint64_t enqueue_sequence_ = 0;
};

}  // namespace xllm
