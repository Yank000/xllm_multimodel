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

#include "prism_dispatcher.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace xllm {
namespace {
constexpr size_t kSkipModelThreshold = 10;
}  // namespace

PrismDispatcher& PrismDispatcher::instance() {
  static PrismDispatcher dispatcher;
  return dispatcher;
}

double PrismDispatcher::calculate_priority_value(double arrival_seconds,
                                                 int32_t slo_ms,
                                                 size_t prompt_len) {
  const double profiled_prefill_seconds =
      std::clamp(static_cast<double>(prompt_len) * 0.5 / 1024.0, 0.2, 2.0);
  return arrival_seconds + static_cast<double>(std::max(0, slo_ms)) / 1000.0 -
         profiled_prefill_seconds;
}

void PrismDispatcher::enqueue_request(const std::string& model_name,
                                      int32_t slo_ms,
                                      size_t prompt_len,
                                      std::function<void()> dispatch_fn) {
  std::vector<std::function<void()>> admitted_dispatches;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto request = std::make_shared<RequestWrapper>();
    request->model_name = model_name;
    request->sequence = enqueue_sequence_++;
    request->priority_value = calculate_priority_value(
        absl::ToDoubleSeconds(absl::Now() - absl::UnixEpoch()), slo_ms, prompt_len);
    request->dispatch_fn = std::move(dispatch_fn);
    queue_.emplace_back(std::move(request));
    admitted_dispatches = admit_requests_locked();
  }
  for (auto& dispatch : admitted_dispatches) {
    dispatch();
  }
}

void PrismDispatcher::mark_request_finished(const std::string& model_name) {
  std::vector<std::function<void()>> admitted_dispatches;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = inflight_by_model_.find(model_name);
    if (it == inflight_by_model_.end() || it->second == 0) {
      LOG(WARNING) << "PRISM inflight counter underflow for model " << model_name;
      return;
    }
    --it->second;
    admitted_dispatches = admit_requests_locked();
  }
  for (auto& dispatch : admitted_dispatches) {
    dispatch();
  }
}

std::vector<std::function<void()>> PrismDispatcher::admit_requests_locked() {
  std::vector<std::function<void()>> admitted;
  while (true) {
    auto best_it = queue_.end();
    double best_priority = std::numeric_limits<double>::max();
    uint64_t best_sequence = std::numeric_limits<uint64_t>::max();

    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
      const auto& request = *it;
      if (inflight_by_model_[request->model_name] >= kSkipModelThreshold) {
        continue;
      }
      if (request->priority_value < best_priority ||
          (request->priority_value == best_priority &&
           request->sequence < best_sequence)) {
        best_it = it;
        best_priority = request->priority_value;
        best_sequence = request->sequence;
      }
    }

    if (best_it == queue_.end()) {
      break;
    }

    auto request = *best_it;
    queue_.erase(best_it);
    ++inflight_by_model_[request->model_name];
    admitted.emplace_back(std::move(request->dispatch_fn));
  }
  return admitted;
}

void PrismDispatcher::reset_for_test() {
  std::lock_guard<std::mutex> guard(mutex_);
  queue_.clear();
  inflight_by_model_.clear();
  enqueue_sequence_ = 0;
}

size_t PrismDispatcher::queued_size_for_test() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return queue_.size();
}

size_t PrismDispatcher::inflight_for_test(const std::string& model_name) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = inflight_by_model_.find(model_name);
  if (it == inflight_by_model_.end()) {
    return 0;
  }
  return it->second;
}

}  // namespace xllm
