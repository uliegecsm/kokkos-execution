#include "gtest/gtest.h"

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c Kokkos::View
 * -------------------------------------
 *
 * This group of tests check the behavior of @c Kokkos::View.
 *
 * The tests can be found in @ref tests/kokkos/test_view.cpp.
 */

namespace Tests {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(ViewTest)
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<AllocateDataEvent, DeallocateDataEvent, BeginFenceEvent, BeginParallelForEvent>;
};

TEST_F(TEST_CATEGORY(ViewTest), configuration) {
#if defined(KOKKOS_ENABLE_IMPL_ASYNC)
    Kokkos::printf("KOKKOS_ENABLE_IMPL_ASYNC is defined.\n");

    static_assert(
        TEST_EXECUTION_SPACE::supports_async_dispatch()
        == Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>);
#else
    Kokkos::printf("KOKKOS_ENABLE_IMPL_ASYNC is NOT defined.\n");
#endif
}

//! @test Allocate and deallocate a @c Kokkos::View of a trivially constructible and destructible type.
TEST_F(TEST_CATEGORY(ViewTest), allocate_and_deallocate_trivial) {
    using view_t = Kokkos::View<int, TEST_EXECUTION_SPACE>;

    const auto recorded_events = recorder_listener_t::record([&]() {
        [[maybe_unused]]
        const view_t data(Kokkos::view_alloc(exec, "my trivial data"));
    });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_ALLOCATE_DATA("my trivial data"), MATCHER_FOR_ALLOCATE_DATA("my trivial data")));
}

//! @test Allocate and deallocate a @c Kokkos::View of a non-trivially constructible nor destructible type.
TEST_F(TEST_CATEGORY(ViewTest), allocate_and_deallocate_non_trivial) {
    struct NonTrivial {
        KOKKOS_FUNCTION
        NonTrivial() noexcept
            : m_value(42) {
        }

        KOKKOS_FUNCTION
        ~NonTrivial() noexcept {
        }

        int m_value;
    };

    static_assert(!std::is_trivially_default_constructible_v<NonTrivial>);
    static_assert(!std::is_trivially_destructible_v<NonTrivial>);

    using view_t = Kokkos::View<NonTrivial[4], TEST_EXECUTION_SPACE>;

    const auto recorded_events = recorder_listener_t::record([&]() {
        [[maybe_unused]]
        const view_t data(Kokkos::view_alloc(exec, "my non-trivial data"));
    });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_ALLOCATE_DATA("my non-trivial data"),
            MATCHER_FOR_BEGIN_PFOR(exec, "Kokkos::View::initialization [my non-trivial data]"),
            MATCHER_FOR_BEGIN_PFOR(exec, "Kokkos::View::destruction [my non-trivial data]"),
            MATCHER_FOR_ALLOCATE_DATA("my non-trivial data")));
}

} // namespace Tests
