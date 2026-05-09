#include "prism_dispatcher.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>

namespace xllm {

TEST(PrismDispatcherTest, PriorityFormulaPrefersLongPromptWhenOthersSame) {
  const double arrival_seconds = 1000.0;
  const int32_t slo_ms = 1000;
  const double short_prompt = PrismDispatcher::calculate_priority_value(
      arrival_seconds, slo_ms, /*prompt_len=*/10);
  const double long_prompt = PrismDispatcher::calculate_priority_value(
      arrival_seconds, slo_ms, /*prompt_len=*/4096);
  EXPECT_LT(long_prompt, short_prompt);
}

TEST(PrismDispatcherTest, SkipModelWhenInflightReachesThreshold) {
  auto& dispatcher = PrismDispatcher::instance();
  dispatcher.reset_for_test();

  std::atomic<int> admitted_a{0};
  std::atomic<int> admitted_b{0};

  for (size_t i = 0; i < 10; ++i) {
    dispatcher.enqueue_request(
        "model-a", 1000, 32, [&admitted_a]() {
          ++admitted_a;
        });
  }

  dispatcher.enqueue_request(
      "model-a", 1000, 32, [&admitted_a]() {
        ++admitted_a;
      });
  dispatcher.enqueue_request(
      "model-b", 1000, 32, [&admitted_b]() {
        ++admitted_b;
      });

  EXPECT_EQ(admitted_a.load(), 10);
  EXPECT_EQ(admitted_b.load(), 1);
  EXPECT_EQ(dispatcher.queued_size_for_test(), 1);
  EXPECT_EQ(dispatcher.inflight_for_test("model-a"), 10);
  EXPECT_EQ(dispatcher.inflight_for_test("model-b"), 1);

  dispatcher.mark_request_finished("model-a");
  EXPECT_EQ(admitted_a.load(), 11);
  EXPECT_EQ(dispatcher.queued_size_for_test(), 0);
}

}  // namespace xllm
