#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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

void process_events();
void insert_events(void* ptr, StreamType stream);

void* my_custom_alloc(size_t size, int device, StreamType stream) {
  (void)stream;
  void* ptr = nullptr;
  if (size <= 0 || device < 0) return nullptr;
  std::lock_guard<std::mutex> lock(mtx);

  process_events();
  bool res = XTensorAllocator::get_instance().allocate_activation(ptr, size);
  if (!res) {
    fprintf(stderr,
            "[custom alloc] XTensorAllocator::allocate_activation failed\n");
    return nullptr;
  }
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

      bool res = XTensorAllocator::get_instance().deallocate_activation(ptr);
      if (!res) {
        LOG(ERROR) << "deallocate_activation failed for ptr=" << ptr;
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
