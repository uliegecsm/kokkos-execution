#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/utils/ThrowsWhenCopied.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c sync_wait by @c Kokkos::Experimental::ExecutionSpaceContext
 * -------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c sync_wait.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_sync_wait.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using ExecutionSpaceContextTest = impl::ExecutionSpaceContextTest<execution_space>;

/**
 * @test Ensure that @c sync_wait is properly customized.
 *
 * Improperly customized @c sync_wait should result in a missing synchronization.
 */
TEST_F(ExecutionSpaceContextTest, sync_wait)
{
    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler());

    Kokkos::utils::callbacks::Manager::initialize();

    ASSERT_THAT(
        Kokkos::utils::callbacks::RecorderListener<Kokkos::utils::callbacks::BeginFenceEvent>::record([chain = std::move(chain)] () mutable {
            const auto value = ::stdexec::sync_wait(std::move(chain));
            static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
            ASSERT_TRUE(value.has_value());
        }),
        ::testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")))
    );

    Kokkos::utils::callbacks::Manager::finalize();
}

//! @test Check that @ref Kokkos::Experimental::details::execution_space::SyncWait properly rethrows if needed.
TEST_F(ExecutionSpaceContextTest, rethrows)
{
    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::then(::tests::utils::ThrowsWhenCopied{});

    ASSERT_THAT(
        ::tests::utils::MutableMoveToSyncWait{.sndr = std::move(chain)},
        testing::ThrowsMessage<std::runtime_error>(testing::StrEq("ThrowsWhenCopied: Throwing in copy constructor!"))
    );
}

} // namespace tests::kokkos_ext
