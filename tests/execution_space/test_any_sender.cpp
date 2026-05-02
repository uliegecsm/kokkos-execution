#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/any_sender_of.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#if defined(KOKKOS_COMPILER_GNU) && KOKKOS_COMPILER_GNU == 1520
#    define KOKKOS_EXECUTION_IMPL_OPSTATE_IMMOVABLE_FIX
#endif
#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Type erased senders and @c Kokkos::Execution::ExecutionSpaceContext
 * -------------------------------------------------------------------
 *
 * This group of tests check that type erased senders are usable with
 * @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * The key design point is that the type erased sender no longer knows the concrete sender type,
 * so the only way to support environment‑based customization is via a fixed set of queries that
 * the erasure wrapper promises to forward. See:
 *  - https://github.com/NVIDIA/stdexec/blob/fa05bc3c93d85c22e8fd987c3b96412a9980f183/include/exec/any_sender_of.hpp#L1310
 *  - https://github.com/NVIDIA/stdexec/blob/fa05bc3c93d85c22e8fd987c3b96412a9980f183/include/exec/any_sender_of.hpp#L1353
 *
 * The tests can be found in @ref tests/execution_space/test_any_sender.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class AnySenderTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<EventDiscardMatcher<TEST_EXECUTION_SPACE>, BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check when synchronization happens.
TEST_F(AnySenderTest, then) {
    using completion_signatures_t =
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>;
    using any_receiver_t = exec::any_receiver<completion_signatures_t>;
    using any_sender_t = exec::any_sender<any_receiver_t>;

    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, any_sender_t>,
                  stdexec::indeterminate_domain<>
    >);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, any_sender_t, stdexec::env<>>,
                  stdexec::default_domain
    >);

    static_assert(std::same_as<
                  std::invoke_result_t<stdexec::get_completion_signatures_t, any_sender_t, stdexec::env<>>,
                  completion_signatures_t
    >);

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    any_sender_t chain = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data) | THEN_INCREMENT(data);

    auto continues_on = std::move(chain) | stdexec::continues_on(esc.get_scheduler());

    static_assert(std::same_as<
                  stdexec::__demangle_t<decltype(continues_on)>,
                  Tests::Utils::basic_sender_t<
                      stdexec::continues_on_t,
                      typename AnySenderTest::scheduler_t,
                      Tests::Utils::basic_sender_t<stdexec::schedule_from_t, stdexec::__, any_sender_t>
                  >
    >);

    any_sender_t extended_chain = std::move(continues_on) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(extended_chain)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch"))));

    ASSERT_EQ(data(), 3);
}

} // namespace Tests::ExecutionSpaceImpl
