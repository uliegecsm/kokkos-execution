#include "tests/kokkos_ext/execution_space/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c sync_wait by @c Kokkos::Experimental::ExecutionSpaceContext
 * -------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c sync_wait.
 *
 * The tests can be found in @ref kokkos_ext/execution_space/test_sync_wait.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using ExecutionSpaceContextTest = impl::ExecutionSpaceContextTest<execution_space>;

/**
 * @test Ensure that @c sync_wait is properly customized.
 *
 * Improperly customized @c sync_wait should result in a missing synchronization.
 *
 * @todo Use some @c Kokkos tools hook to ensure that proper fencing happens instead of relying on captured output.
 */
TEST_F(ExecutionSpaceContextTest, sync_wait)
{
    const context_t esc{*exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler());

    ::testing::internal::CaptureStdout();

    const auto value = ::stdexec::sync_wait(std::move(chain));

    EXPECT_EQ(::testing::internal::GetCapturedStdout(), std::format("SyncWaitReceiver: fencing {} ({})\n", Kokkos::Impl::TypeInfo<execution_space>::name(), exec->impl_instance_id()));

    static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);

    ASSERT_TRUE(value.has_value());
}

} // namespace tests::kokkos_ext
