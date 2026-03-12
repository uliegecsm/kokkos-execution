#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"

/**
 * @addtogroup unittests
 *
 * Custom @c scoped_region for @c Kokkos::Execution::ExecutionSpaceContext
 * -----------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly defines
 * a custom algorithm @c scoped_region.
 *
 * The tests can be found in @ref tests/execution_space/test_scoped_region.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class ScopedRegionTest
    : public Tests::Utils::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<BeginFenceEvent, BeginParallelForEvent, PushRegionEvent, PopRegionEvent>;
};

/**
 * @test Check that @ref Kokkos::Profiling::scoped_region works as intended.
 *
 * The push/pop events and the preceding fences must be placed appropriately.
 */
TEST_F(ScopedRegionTest, many) {
    const view_s_t data(Kokkos::view_alloc("data - shared space", exec));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler())
               | Kokkos::Profiling::scoped_region(
                     "the name of my nice scoped region", THEN_INCREMENT(data) | THEN_INCREMENT(data));

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "push")),
            MATCHER_FOR_PUSH_REGION("the name of my nice scoped region"),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "pop")),
            MATCHER_FOR_POP_REGION(),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
}

} // namespace Tests::ExecutionSpaceImpl
