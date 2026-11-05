#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/check_sync_wait.hpp"
#include "tests/utils/execution_space_context.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c sync_wait by @c Kokkos::Execution::ExecutionSpaceContext
 * ----------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c sync_wait.
 *
 * The tests can be found in @ref tests/execution_space/test_sync_wait.cpp.
 */

namespace Tests {

namespace Utils {
STDEXEC_PP_EVAL(
    INSTANTIATE_TYPED_TEST_SUITE_P,
    TEST_CATEGORY(SyncWaitTestExecutionSpaceInstantiation),
    TEST_CATEGORY(SyncWaitTest),
    ::testing::Types<Kokkos::Execution::ExecutionSpaceContext<TEST_EXECUTION_SPACE>>);
} // namespace Utils

namespace ExecutionSpaceImpl {

class TEST_CATEGORY(SyncWaitTest)
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE> { };

//! @test Check whether the sender can be nothrow applied.
static_assert(Tests::Utils::check_nothrow_apply_sender<
              Kokkos::Execution::ExecutionSpaceImpl::Domain,
              typename TEST_CATEGORY(SyncWaitTest)::schedule_sender_t
>());

} // namespace ExecutionSpaceImpl

} // namespace Tests
