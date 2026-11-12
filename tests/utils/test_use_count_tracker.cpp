#include <latch>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/utils/functors/use_count_tracker.hpp"

/**
 * @addtogroup unittests
 *
 * Check @c Tests::Utils::Functors::UseCountTracker
 * ------------------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::Functors::UseCountTracker and related.
 *
 * The tests can be found in @ref tests/utils/test_use_count_tracker.cpp.
 */

namespace Tests {

class UseCountTrackerTest : public testing::Test {
   protected:
    Utils::Functors::UseCountTrackerSource source;
};

//! @test Check that @ref Tests::Utils::Functors::UseCountTracker call operator is @c noexcept.
consteval bool test_call_operator_is_noexcept() {
    static_assert(noexcept(std::declval<const Utils::Functors::UseCountTracker&>()(1, 2.0, "x")));

    return true;
}

static_assert(test_call_operator_is_noexcept());

//! @test Check that @ref Tests::Utils::Functors::UseCountTrackerSource::use_count and @ref Tests::Utils::Functors::UseCountTrackerSource::high_water start at zero.
TEST_F(UseCountTrackerTest, source_use_count_and_high_water_start_at_zero) {
    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(source.high_water(), 0);
}

//! @test The default-constructed @ref Tests::Utils::Functors::UseCountTracker is untracked.
TEST_F(UseCountTrackerTest, default_constructed_is_untracked) {
    static_assert(std::is_nothrow_constructible_v<Utils::Functors::UseCountTracker>);

    const Utils::Functors::UseCountTracker l_a;
    const Utils::Functors::UseCountTracker l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_THAT(l_a.m_state.get(), testing::IsNull());
    EXPECT_THAT(l_b.m_state.get(), testing::IsNull());

    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(source.high_water(), 0);
}

//! @test Constructing a @ref Tests::Utils::Functors::UseCountTracker from a @ref Tests::Utils::Functors::UseCountTrackerSource increases the use count by one.
TEST_F(UseCountTrackerTest, construction_from_source_counts_once) {
    const Utils::Functors::UseCountTracker l_a{source};

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);
}

//! @test Copy-construction increments the use count by one.
TEST_F(UseCountTrackerTest, copy_construction_increments) {
    const Utils::Functors::UseCountTracker l_a{source};

    const Utils::Functors::UseCountTracker l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(source.use_count(), 2);
    EXPECT_EQ(source.high_water(), 2);
}

//! @test Destruction decreases the use count by one, but leaves the high water untouched.
TEST_F(UseCountTrackerTest, destruction_decreases_but_keeps_high_water) {
    const Utils::Functors::UseCountTracker l_a{source};

    {
        const Utils::Functors::UseCountTracker l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)
        const Utils::Functors::UseCountTracker l_c = l_b; // NOLINT(performance-unnecessary-copy-initialization)

        EXPECT_EQ(source.use_count(), 3);
        EXPECT_EQ(source.high_water(), 3);
    }

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 3);
}

//! @test Copy-assign into an untracked @ref Tests::Utils::Functors::UseCountTracker increases the use count by one.
TEST_F(UseCountTrackerTest, copy_assignment_into_untracked_increments) {
    const Utils::Functors::UseCountTracker l_a{source};

    Utils::Functors::UseCountTracker l_b;
    l_b = l_a;

    EXPECT_EQ(source.use_count(), 2);
    EXPECT_EQ(source.high_water(), 2);
}

//! @test Copy-assign of a tracked @ref Tests::Utils::Functors::UseCountTracker into another tracked @ref Tests::Utils::Functors::UseCountTracker is neutral.
TEST_F(UseCountTrackerTest, copy_assignment_between_trackers_of_one_source_is_neutral) {
    const Utils::Functors::UseCountTracker l_a{source};

    Utils::Functors::UseCountTracker l_b{source};

    ASSERT_EQ(source.use_count(), 2);
    ASSERT_EQ(source.high_water(), 2);

    l_b = l_a;

    EXPECT_EQ(source.use_count(), 2);
    EXPECT_EQ(source.high_water(), 2);
}

//! @test Copy-assign from different @ref Tests::Utils::Functors::UseCountTrackerSource changes the use count of both sources.
TEST_F(UseCountTrackerTest, copy_assignment_across_sources_moves_the_reference) {
    const Utils::Functors::UseCountTrackerSource other_source;

    Utils::Functors::UseCountTracker l_x{source};

    const Utils::Functors::UseCountTracker l_y{other_source};

    l_x = l_y;

    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(source.high_water(), 1);

    EXPECT_EQ(other_source.use_count(), 2);
    EXPECT_EQ(other_source.high_water(), 2);
}

//! @test Move-construction moves the @ref Tests::Utils::Functors::UseCountTracker::m_state and does not change the use count, nor the high water.
TEST_F(UseCountTrackerTest, move_construction_transfers_without_counting) {
    Utils::Functors::UseCountTracker l_a{source};

    const Utils::Functors::UseCountTracker l_b(std::move(l_a));

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);

    EXPECT_THAT(l_a.m_state.get(), testing::IsNull()); // NOLINT(bugprone-use-after-move)
    EXPECT_THAT(l_b.m_state.get(), testing::Not(testing::IsNull()));

    const Utils::Functors::UseCountTracker l_c = l_a; // NOLINT(bugprone-use-after-move)

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);

    EXPECT_THAT(l_c.m_state.get(), testing::IsNull());
}

//! @test Move-assignment moves the @ref Tests::Utils::Functors::UseCountTracker::m_state and does not change the use count, nor the high water.
TEST_F(UseCountTrackerTest, move_assignment_transfers_without_counting) {
    const Utils::Functors::UseCountTrackerSource other_source;

    {
        Utils::Functors::UseCountTracker l_a{source};
        Utils::Functors::UseCountTracker l_b{other_source};

        l_a = std::move(l_b);

        EXPECT_THAT(l_b.m_state.get(), testing::IsNull()); // NOLINT(bugprone-use-after-move)
        EXPECT_THAT(l_a.m_state.get(), testing::Not(testing::IsNull()));

        EXPECT_EQ(source.use_count(), 0);
        EXPECT_EQ(other_source.use_count(), 1);

        EXPECT_EQ(source.high_water(), 1);
        EXPECT_EQ(other_source.high_water(), 1);
    }

    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(other_source.use_count(), 0);
}

//! @test Using @ref Tests::Utils::Functors::UseCountTrackerSource::reset_high_water restarts from the current use count.
TEST_F(UseCountTrackerTest, reset_high_water_restarts_from_the_current_count) {
    const Utils::Functors::UseCountTracker l_a{source};

    {
        const Utils::Functors::UseCountTracker l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)
        const Utils::Functors::UseCountTracker l_c = l_a; // NOLINT(performance-unnecessary-copy-initialization)
    }

    ASSERT_EQ(source.high_water(), 3);
    source.reset_high_water();

    EXPECT_EQ(source.high_water(), 1);

    const Utils::Functors::UseCountTracker l_d = l_a; // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(source.high_water(), 2);
}

//! GCC defines @c __SANITIZE_THREAD__. Clang exposes @c __has_feature(thread_sanitizer).
#if defined(__SANITIZE_THREAD__)
#    define KOKKOS_EXECUTION_TSAN_DETECTED 1 // NOLINT(cppcoreguidelines-macro-usage)
#elif defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define KOKKOS_EXECUTION_TSAN_DETECTED 1
#    endif
#endif

#ifdef KOKKOS_EXECUTION_TSAN_DETECTED
#    define KOKKOS_EXECUTION_SKIP_IF_TSAN(reason) GTEST_SKIP() << reason // NOLINT(cppcoreguidelines-macro-usage)
#else
#    define KOKKOS_EXECUTION_SKIP_IF_TSAN(...)
#endif

//! @test @ref Tests::Utils::Functors::UseCountTracker is thread-safe.
TEST_F(UseCountTrackerTest, concurrent_copies_reach_the_expected_high_water) {
    //! See https://github.com/kokkos/kokkos/pull/9773.
    KOKKOS_EXECUTION_SKIP_IF_TSAN("Kokkos::Impl::HostSharedPtr::use_count generates TSan warnings.");

    constexpr unsigned short int num_threads = 18;

    const Utils::Functors::UseCountTracker l_a{source};

    std::latch all_copied{num_threads};
    std::latch release{1};

    {
        std::array<std::jthread, num_threads> threads;
        for (auto& thread: threads)
            thread = std::jthread([&] {
                const Utils::Functors::UseCountTracker l_c = l_a; // NOLINT(performance-unnecessary-copy-initialization)
                all_copied.count_down();
                release.wait();
            });
        all_copied.wait();

        EXPECT_EQ(source.use_count(), num_threads + 1);
        EXPECT_EQ(source.high_water(), num_threads + 1);

        release.count_down();
    }

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), num_threads + 1);
}

} // namespace Tests
