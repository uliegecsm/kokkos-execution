#include <latch>
#include <thread>

#include "gtest/gtest.h"

#include "tests/utils/functors/lifetime.hpp"

/**
 * @addtogroup unittests
 *
 * Check @c Tests::Utils::Functors::Lifetime
 * -----------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::Functors::Lifetime and related.
 *
 * The tests can be found in @ref tests/utils/test_lifetime.cpp.
 */

namespace Tests {

class LifetimeTest : public testing::Test {
   protected:
    Utils::Functors::LifetimeSource source;
};

//! @test Check that @ref Tests::Utils::Functors::Lifetime call operator is @c noexcept.
consteval bool test_call_operator_is_noexcept() {
    static_assert(noexcept(std::declval<const Utils::Functors::Lifetime&>()(1, 2.0, "x")));

    return true;
}

static_assert(test_call_operator_is_noexcept());

//! @test Check that @ref Tests::Utils::Functors::LifetimeSource::use_count and @ref Tests::Utils::Functors::LifetimeSource::high_water start at zero.
TEST_F(LifetimeTest, source_use_count_and_high_water_start_at_zero) {
    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(source.high_water(), 0);
}

//! @test The default-constructed @ref Tests::Utils::Functors::Lifetime is untracked.
TEST_F(LifetimeTest, default_constructed_is_untracked) {
    static_assert(std::is_nothrow_constructible_v<Utils::Functors::Lifetime>);

    const Utils::Functors::Lifetime l_a;
    const Utils::Functors::Lifetime l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_FALSE(l_a.tracker.has_record());
    EXPECT_FALSE(l_b.tracker.has_record());

    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(source.high_water(), 0);
}

//! @test Constructing a @ref Tests::Utils::Functors::Lifetime from a @ref Tests::Utils::Functors::LifetimeSource increases the use count by one.
TEST_F(LifetimeTest, construction_from_source_counts_once) {
    const Utils::Functors::Lifetime l_a{source};

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);
}

//! @test Copy-construction increments the use count by one.
TEST_F(LifetimeTest, copy_construction_increments) {
    const Utils::Functors::Lifetime l_a{source};

    const Utils::Functors::Lifetime l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(source.use_count(), 2);
    EXPECT_EQ(source.high_water(), 2);
}

//! @test Destruction decreases the use count by one, but leaves the high water untouched.
TEST_F(LifetimeTest, destruction_decreases_but_keeps_high_water) {
    const Utils::Functors::Lifetime l_a{source};
    {
        const Utils::Functors::Lifetime l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)
        const Utils::Functors::Lifetime l_c = l_b; // NOLINT(performance-unnecessary-copy-initialization)

        EXPECT_EQ(source.use_count(), 3);
        EXPECT_EQ(source.high_water(), 3);
    }

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 3);
}

//! @test Copy-assign into an untracked @ref Tests::Utils::Functors::Lifetime increases the use count by one.
TEST_F(LifetimeTest, copy_assignment_into_untracked_increments) {
    const Utils::Functors::Lifetime l_a{source};

    Utils::Functors::Lifetime l_b;
    l_b = l_a;

    EXPECT_EQ(source.use_count(), 2);
    EXPECT_EQ(source.high_water(), 2);
}

//! @test Copy-assign of a tracked @ref Tests::Utils::Functors::Lifetime into another tracked @ref Tests::Utils::Functors::Lifetime is neutral.
TEST_F(LifetimeTest, copy_assignment_between_trackers_of_one_source_is_neutral) {
    const Utils::Functors::Lifetime l_a{source};

    Utils::Functors::Lifetime l_b{source};

    ASSERT_EQ(source.use_count(), 2);
    ASSERT_EQ(source.high_water(), 2);

    l_b = l_a;

    EXPECT_EQ(source.use_count(), 2);
    EXPECT_EQ(source.high_water(), 2);
}

//! @test Copy-assign from different @ref Tests::Utils::Functors::LifetimeSource changes the use count of both sources.
TEST_F(LifetimeTest, copy_assignment_across_sources_moves_the_reference) {
    Utils::Functors::LifetimeSource other_source;

    Utils::Functors::Lifetime l_x{source};

    const Utils::Functors::Lifetime l_y{other_source};

    l_x = l_y;

    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(source.high_water(), 1);

    EXPECT_EQ(other_source.use_count(), 2);
    EXPECT_EQ(other_source.high_water(), 2);
}

//! @test Moving moves the @ref Tests::Utils::Functors::Lifetime::tracker and does not change the use count, nor the high water.
TEST_F(LifetimeTest, move_construction_transfers_without_counting) {
    Utils::Functors::Lifetime l_a{source};

    const Utils::Functors::Lifetime l_b(std::move(l_a));

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);

    EXPECT_FALSE(l_a.tracker.has_record()); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(l_b.tracker.has_record());

    const Utils::Functors::Lifetime l_c =
        l_a; // NOLINT(bugprone-use-after-move): the moved-from tracker must be untracked

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);

    EXPECT_FALSE(l_c.tracker.has_record());
}

//! @test See https://github.com/kokkos/kokkos/pull/9769. It should be fixed upstream, and does not impact our use case of @ref Tests::Utils::Functors::Lifetime.
TEST_F(LifetimeTest, move_assignment_swaps_ownership) {
    Utils::Functors::LifetimeSource other_source;

    {
        Utils::Functors::Lifetime l_x{source};
        Utils::Functors::Lifetime l_y{other_source};

        l_x = std::move(l_y);

        EXPECT_TRUE(l_x.tracker.has_record());
        EXPECT_TRUE(l_y.tracker.has_record()); // NOLINT(bugprone-use-after-move)

        EXPECT_EQ(source.use_count(), 1);
        EXPECT_EQ(other_source.use_count(), 1);

        EXPECT_EQ(source.high_water(), 1);
        EXPECT_EQ(other_source.high_water(), 1);
    }

    EXPECT_EQ(source.use_count(), 0);
    EXPECT_EQ(other_source.use_count(), 0);
}

//! @test Using @ref Tests::Utils::Functors::LifetimeSource::reset_high_water restarts from the current use count.
TEST_F(LifetimeTest, reset_high_water_restarts_from_the_current_count) {
    const Utils::Functors::Lifetime l_a{source};

    {
        const Utils::Functors::Lifetime l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)
        const Utils::Functors::Lifetime l_c = l_a; // NOLINT(performance-unnecessary-copy-initialization)
    }

    ASSERT_EQ(source.high_water(), 3);
    source.reset_high_water();

    EXPECT_EQ(source.high_water(), 1);

    const Utils::Functors::Lifetime l_d = l_a; // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(source.high_water(), 2);
}

//! @test Copies made with tracking disabled are not taken into account.
TEST_F(LifetimeTest, copies_made_with_tracking_disabled_are_not_counted) {
    const Utils::Functors::Lifetime l_a{source};

    {
        const Kokkos::Impl::SharedAllocationDisableTrackingGuard guard;
        const Utils::Functors::Lifetime l_b = l_a; // NOLINT(performance-unnecessary-copy-initialization)

        EXPECT_EQ(source.use_count(), 1);
        EXPECT_EQ(source.high_water(), 1);
    }

    EXPECT_EQ(source.use_count(), 1);
    EXPECT_EQ(source.high_water(), 1);
}

//! @test @ref Tests::Utils::Functors::Lifetime is thread-safe.
TEST_F(LifetimeTest, concurrent_copies_reach_the_expected_high_water) {
    constexpr unsigned short int num_threads = 18;

    const Utils::Functors::Lifetime l_a{source};

    std::latch all_copied{num_threads};
    std::latch release{1};

    {
        std::array<std::jthread, num_threads> threads;
        for (auto& thread: threads)
            thread = std::jthread([&] {
                const Utils::Functors::Lifetime l_c = l_a; // NOLINT(performance-unnecessary-copy-initialization)
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
