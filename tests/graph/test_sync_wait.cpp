#include "kokkos-execution/graph.hpp"

#include "tests/utils/check_sync_wait.hpp"
#include "tests/utils/graph_context.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c stdexec::sync_wait by @c Kokkos::Execution::GraphContext
 * ----------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly customizes
 * @c stdexec::sync_wait.
 *
 * The tests can be found in @ref tests/graph/test_sync_wait.cpp.
 */

namespace Tests {

namespace Utils {
STDEXEC_PP_EVAL(
    INSTANTIATE_TYPED_TEST_SUITE_P,
    TEST_CATEGORY(SyncWaitTestGraphInstantiation),
    TEST_CATEGORY(SyncWaitTest),
    ::testing::Types<Kokkos::Execution::GraphContext<TEST_EXECUTION_SPACE>>);
} // namespace Utils

namespace GraphImpl {

class TEST_CATEGORY(SyncWaitTest)
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE> { };

//! @test Check whether the sender can be nothrow applied.
static_assert(Tests::Utils::check_nothrow_apply_sender<
              Kokkos::Execution::GraphImpl::Domain,
              typename TEST_CATEGORY(SyncWaitTest)::schedule_sender_t
>());

} // namespace GraphImpl

} // namespace Tests
