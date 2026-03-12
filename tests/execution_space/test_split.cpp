#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsign-compare")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/split.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"

/**
 * @addtogroup unittests
 *
 * Both @c experimental::execution::split and @c stdexec::when_all are supported by @c Kokkos::Execution::ExecutionSpaceContext
 * ----------------------------------------------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly works with both
 * @c experimental::execution::split and @c stdexec::when_all.
 *
 * The tests can be found in @ref tests/execution_space/test_split.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class SplitTest
    : public Tests::Utils::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Use @c experimental::execution::split and @c stdexec::sync_wait right after.
TEST_F(SplitTest, split_and_sync_wait) {
    const context_t esc{exec};

    stdexec::sender auto chain = stdexec::schedule(esc.get_scheduler()) | experimental::execution::split();

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); }),
        testing::IsEmpty());
}

/**
 * @test Each branch of a @c stdexec::when_all that executes on @ref Kokkos::Execution::ExecutionSpaceContext must synchronize before calling
 *       @c stdexec::set_value of the "end of the branch receiver".
 */
TEST_F(SplitTest, within) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    experimental::execution::static_thread_pool pool{4};
    const context_t esc{exec};

    stdexec::sender auto fork = stdexec::schedule(pool.get_scheduler()) | experimental::execution::split();

    auto branch_a = fork | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data)
                  | THEN_INCREMENT_ATOMIC(data);
    auto branch_b = fork | stdexec::continues_on(pool.get_scheduler()) | THEN_INCREMENT_ATOMIC(data);
    auto branch_c = std::move(fork) | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data)
                  | THEN_INCREMENT_ATOMIC(data);

    auto chain = stdexec::when_all(std::move(branch_a), std::move(branch_b), std::move(branch_c))
               | stdexec::then([&data]() {
                     if (data() != 5)
                         Kokkos::abort("Synchronization issue.");
                 });

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    /// Each branch may be executed by a distinct host thread. However, the callback manager is not thread safe.
    /// So ordering of the event is not guaranteed.
    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); }),
        testing::UnorderedElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation"))));

    ASSERT_EQ(data(), 5);
}

} // namespace Tests::ExecutionSpaceImpl
