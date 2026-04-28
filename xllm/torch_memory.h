#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(USE_NPU)
#include "acl/acl.h"
#elif defined(USE_CUDA)
#include <cuda_runtime_api.h>
#else
#error "torch_memory.h requires USE_NPU or USE_CUDA"
#endif

#include "core/framework/xtensor/xtensor_allocator.h"

using namespace xllm;

std::mutex mtx;

enum class EventQueryState {
  kReady,
  kNotReady,
  kError,
};

#if defined(USE_NPU)
using StreamType = aclrtStream;
using EventType = aclrtEvent;
using ErrorType = aclError;
constexpr ErrorType kEventSuccess = ACL_ERROR_NONE;
#elif defined(USE_CUDA)
using StreamType = cudaStream_t;
using EventType = cudaEvent_t;
using ErrorType = cudaError_t;
constexpr ErrorType kEventSuccess = cudaSuccess;
constexpr ErrorType kEventNotReady = cudaErrorNotReady;
#endif

inline bool EventSuccess(ErrorType err) { return err == kEventSuccess; }

inline std::string EventError(ErrorType err) {
#if defined(USE_NPU)
  return std::to_string(static_cast<int>(err));
#elif defined(USE_CUDA)
  return cudaGetErrorString(err);
#endif
}

inline ErrorType EventCreate(EventType* event) {
#if defined(USE_NPU)
  return aclrtCreateEvent(event);
#elif defined(USE_CUDA)
  return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
#endif
}

inline ErrorType EventRecord(EventType event, StreamType stream) {
#if defined(USE_NPU)
  return aclrtRecordEvent(event, stream);
#elif defined(USE_CUDA)
  return cudaEventRecord(event, stream);
#endif
}

inline ErrorType EventDestroy(EventType event) {
#if defined(USE_NPU)
  return aclrtDestroyEvent(event);
#elif defined(USE_CUDA)
  return cudaEventDestroy(event);
#endif
}

inline EventQueryState EventQuery(EventType event) {
#if defined(USE_NPU)
  aclrtEventRecordedStatus status;
  ErrorType err = aclrtQueryEventStatus(event, &status);
  if (!EventSuccess(err)) {
    LOG(ERROR) << "EventQuery failed: " << EventError(err);
    return EventQueryState::kError;
  }
  return status == ACL_EVENT_RECORDED_STATUS_COMPLETE
             ? EventQueryState::kReady
             : EventQueryState::kNotReady;
#elif defined(USE_CUDA)
  ErrorType err = cudaEventQuery(event);
  if (err == kEventNotReady) {
    return EventQueryState::kNotReady;
  }
  if (!EventSuccess(err)) {
    LOG(ERROR) << "EventQuery failed: " << EventError(err);
    return EventQueryState::kError;
  }
  return EventQueryState::kReady;
#endif
}

ska::flat_hash_map<StreamType, std::deque<std::pair<EventType, void*>>>
    pending_events;

struct CachedActivationBlock {
  void* ptr;
  size_t size;
  std::chrono::steady_clock::time_point timestamp;
};

struct CachedActivationSpan {
  std::vector<CachedActivationBlock> blocks;
};

inline uintptr_t cached_block_start(const CachedActivationBlock& block) {
  return reinterpret_cast<uintptr_t>(block.ptr);
}

inline uintptr_t cached_block_end(const CachedActivationBlock& block) {
  return cached_block_start(block) + block.size;
}

inline uintptr_t cached_span_end(const CachedActivationSpan& span) {
  return cached_block_end(span.blocks.back());
}

struct CachedActivationPool {
  std::map<uintptr_t, CachedActivationSpan> spans;
  std::unordered_map<void*, std::vector<CachedActivationBlock>>
      active_allocations;

  void store(void* ptr, size_t size) {
    release_expired();
    store_block(CachedActivationBlock{
        ptr,
        size,
        std::chrono::steady_clock::now(),
    });
  }

  void store(std::vector<CachedActivationBlock>&& blocks) {
    release_expired();
    auto now = std::chrono::steady_clock::now();
    for (auto& block : blocks) {
      block.timestamp = now;
      store_block(block);
    }
  }

  bool allocate(void*& ptr, size_t size) {
    auto best_span = spans.end();
    size_t best_start = 0;
    size_t best_end = 0;
    size_t best_size = std::numeric_limits<size_t>::max();

    for (auto span_it = spans.begin(); span_it != spans.end(); ++span_it) {
      auto& blocks = span_it->second.blocks;
      size_t start = 0;
      size_t total = 0;
      for (size_t end = 0; end < blocks.size(); ++end) {
        total += blocks[end].size;
        while (start < end && total - blocks[start].size >= size) {
          total -= blocks[start].size;
          ++start;
        }
        if (total >= size && total < best_size) {
          best_span = span_it;
          best_start = start;
          best_end = end + 1;
          best_size = total;
        }
      }
    }

    if (best_span == spans.end()) {
      return false;
    }

    auto blocks = std::vector<CachedActivationBlock>(
        best_span->second.blocks.begin() + best_start,
        best_span->second.blocks.begin() + best_end);
    ptr = blocks.front().ptr;
    remove_range(best_span, best_start, best_end);
    if (blocks.size() > 1) {
      active_allocations.emplace(ptr, std::move(blocks));
    }
    return true;
  }

  bool take_active(void* ptr, std::vector<CachedActivationBlock>& blocks) {
    auto it = active_allocations.find(ptr);
    if (it == active_allocations.end()) {
      return false;
    }
    blocks = std::move(it->second);
    active_allocations.erase(it);
    return true;
  }

 private:
  void store_block(const CachedActivationBlock& block) {
    uintptr_t addr = cached_block_start(block);
    auto next = spans.lower_bound(addr);
    auto prev = next == spans.begin() ? spans.end() : std::prev(next);
    bool merge_prev =
        prev != spans.end() && cached_span_end(prev->second) == addr;
    bool merge_next =
        next != spans.end() &&
        addr + block.size == cached_block_start(next->second.blocks.front());

    if (merge_prev && merge_next) {
      prev->second.blocks.push_back(block);
      prev->second.blocks.insert(prev->second.blocks.end(),
                                 next->second.blocks.begin(),
                                 next->second.blocks.end());
      spans.erase(next);
      return;
    }
    if (merge_prev) {
      prev->second.blocks.push_back(block);
      return;
    }
    if (merge_next) {
      CachedActivationSpan span;
      span.blocks.push_back(block);
      span.blocks.insert(span.blocks.end(),
                         next->second.blocks.begin(),
                         next->second.blocks.end());
      spans.erase(next);
      spans.emplace(addr, std::move(span));
      return;
    }
    CachedActivationSpan span;
    span.blocks.push_back(block);
    spans.emplace(addr, std::move(span));
  }

  void release_expired() {
    auto now = std::chrono::steady_clock::now();
    std::map<uintptr_t, CachedActivationSpan> retained;

    for (auto& entry : spans) {
      std::vector<CachedActivationBlock> current;
      for (auto& block : entry.second.blocks) {
        if (now - block.timestamp > std::chrono::milliseconds(100)) {
          void* ptr = block.ptr;
          XTensorAllocator::get_instance().deallocate_activation(ptr);
          if (!current.empty()) {
            CachedActivationSpan span;
            span.blocks = std::move(current);
            retained.emplace(cached_block_start(span.blocks.front()),
                             std::move(span));
            current.clear();
          }
          continue;
        }
        current.push_back(block);
      }
      if (!current.empty()) {
        CachedActivationSpan span;
        span.blocks = std::move(current);
        retained.emplace(cached_block_start(span.blocks.front()),
                         std::move(span));
      }
    }

    spans.swap(retained);
  }

  void remove_range(std::map<uintptr_t, CachedActivationSpan>::iterator span_it,
                    size_t start,
                    size_t end) {
    auto blocks = std::move(span_it->second.blocks);
    spans.erase(span_it);

    if (start > 0) {
      CachedActivationSpan left;
      left.blocks.assign(blocks.begin(), blocks.begin() + start);
      spans.emplace(cached_block_start(left.blocks.front()), std::move(left));
    }
    if (end < blocks.size()) {
      CachedActivationSpan right;
      right.blocks.assign(blocks.begin() + end, blocks.end());
      spans.emplace(cached_block_start(right.blocks.front()), std::move(right));
    }
  }
};

inline CachedActivationPool& cached_activation_pool() {
  static CachedActivationPool pool;
  return pool;
}

void process_events();
void insert_events(void* ptr, StreamType stream);

void* my_custom_alloc(size_t size, int device, StreamType stream) {
  (void)stream;
  void* ptr = nullptr;
  if (size <= 0 || device < 0) return nullptr;
  std::lock_guard<std::mutex> lock(mtx);

  process_events();
  if (cached_activation_pool().allocate(ptr, size)) {
    // LOG(INFO) << "[custom alloc] allocate from cache pool, ptr=" << ptr
    //           << ", size=" << size;
    return ptr;
  }
  bool res = XTensorAllocator::get_instance().allocate_activation(ptr, size);
  if (!res) {
    fprintf(stderr,
            "[custom alloc] XTensorAllocator::allocate_activation failed\n");
    return nullptr;
  }
  // LOG(INFO) << "[custom alloc] allocate from xtensor allocator, ptr=" << ptr
  //           << ", size=" << size;
  return ptr;
}

void process_events() {
  for (auto it = pending_events.begin(); it != pending_events.end();) {
    while (!it->second.empty()) {
      auto& e = it->second.front();
      EventType event = e.first;
      void* ptr = e.second;

      EventQueryState state = EventQuery(event);
      if (state == EventQueryState::kNotReady) {
        break;
      }
      if (state == EventQueryState::kError) {
        break;
      }

      std::vector<CachedActivationBlock> blocks;
      if (cached_activation_pool().take_active(ptr, blocks)) {
        cached_activation_pool().store(std::move(blocks));
      } else {
        auto alloc_it = XTensorAllocator::get_instance().find_ptr(ptr);
        size_t alloc_size = alloc_it->second;
        cached_activation_pool().store(ptr, alloc_size);
      }
      ErrorType destroy_err = EventDestroy(event);
      if (!EventSuccess(destroy_err)) {
        LOG(ERROR) << "EventDestroy failed: " << EventError(destroy_err);
      }
      it->second.pop_front();
    }
    if (it->second.empty()) {
      it = pending_events.erase(it);
    } else {
      ++it;
    }
  }
}

void my_custom_free(void* ptr, size_t size, int device, StreamType stream) {
  (void)size;
  (void)device;
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx);
  insert_events(ptr, stream);
}

void insert_events(void* ptr, StreamType stream) {
  EventType event;
  ErrorType create_err = EventCreate(&event);
  if (!EventSuccess(create_err)) {
    LOG(ERROR) << "EventCreate failed: " << EventError(create_err);
    void* p = ptr;
    XTensorAllocator::get_instance().deallocate_activation(p);
    return;
  }

  ErrorType record_err = EventRecord(event, stream);
  if (!EventSuccess(record_err)) {
    LOG(ERROR) << "EventRecord failed: " << EventError(record_err);
    ErrorType destroy_err = EventDestroy(event);
    if (!EventSuccess(destroy_err)) {
      LOG(ERROR) << "EventDestroy failed: " << EventError(destroy_err);
    }
    void* p = ptr;
    XTensorAllocator::get_instance().deallocate_activation(p);
    return;
  }

  pending_events[stream].emplace_back(event, ptr);
}
