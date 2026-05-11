#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/tracking_allocator.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Tests::Utils::TrackingAllocator
 * --------------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::TrackingAllocator.
 *
 * The tests can be found in @ref tests/utils/test_tracking_allocator.cpp.
 */

namespace Tests {

//! @test The allocator injected with @c stdexec::write_env is visible *via* @c stdexec::read_env.
TEST(TrackingAllocatorTest, write_read_env) {
    std::atomic<size_t> count = 0;

    stdexec::sender auto sndr =
        stdexec::read_env(stdexec::get_allocator) | stdexec::then([](auto allocator) {
            int value = 42;
            return Tests::Utils::round_trip_allocate(allocator, static_cast<int&>(value));
        })
        | stdexec::write_env(stdexec::prop{stdexec::get_allocator, Tests::Utils::TrackingAllocator<int>{&count}});

    const auto [val] = stdexec::sync_wait(std::move(sndr)).value(); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(val, 42);
    ASSERT_EQ(count, 1);
}

//! @test The allocator is forwarded through @c stdexec::then adaptor because @c stdexec::get_allocator_t is a forwarding query.
TEST(TrackingAllocatorTest, forwarded_through_then) {
    static_assert(stdexec::forwarding_query(stdexec::get_allocator));

    std::atomic<size_t> count = 0;

    stdexec::sender auto sndr =
        stdexec::read_env(stdexec::get_allocator)
        | stdexec::then([](auto allocator) { return Tests::Utils::round_trip_allocate(allocator, 42); })
        | stdexec::then([](auto&& value) { return value; })
        | stdexec::write_env(stdexec::prop{stdexec::get_allocator, Tests::Utils::TrackingAllocator<int>{&count}});

    const auto [val] = stdexec::sync_wait(std::move(sndr)).value(); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(val, 42);
    ASSERT_EQ(count, 1);
}

} // namespace Tests
