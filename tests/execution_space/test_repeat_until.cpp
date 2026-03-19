#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/repeat_until.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/sum_indices.hpp"

/**
 * @addtogroup unittests
 *
 * Use @c Kokkos::Execution::ExecutionSpaceContext with @c experimental:execution::repeat_until
 * --------------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly interacts with
 * @c experimental:execution::repeat_until.
 *
 * The tests can be found in @ref tests/execution_space/test_repeat_until.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class RepeatEffectUntilTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check that @ref Kokkos::Execution::ExecutionSpaceContext can be properly embedded in a @c experimental:execution::repeat_until.
TEST_F(RepeatEffectUntilTest, works) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data) | BULK_SUM_INDICES(2, data);

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable {
            unsigned int guard = 0;
            stdexec::sync_wait(
                experimental::execution::repeat_until(
                    std::move(chain) | stdexec::continues_on(stdexec::inline_scheduler{})
                    | stdexec::then([&guard]() -> bool { return (++guard) >= 3; })));
        }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 6);
}

} // namespace Tests::ExecutionSpaceImpl
