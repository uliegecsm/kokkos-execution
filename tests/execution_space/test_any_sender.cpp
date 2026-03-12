#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "exec/any_sender_of.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/stdexec.hpp"

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

using execution_space = Kokkos::DefaultExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class AnySenderTest
    : public Tests::Utils::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

/**
 * @test In order to get the same synchronization behavior as if using fully typed senders,
 *       type erased senders must advertise the completion domain and scheduler and type erased receivers
 *       must advertise their @ref Kokkos::Execution::ExecutionSpaceImpl::get_exec_t query.
 */
TEST_F(AnySenderTest, then) {
    using any_sender_t = experimental::execution::any_receiver_ref<
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
        Kokkos::Execution::ExecutionSpaceImpl::get_exec
            .signature<Kokkos::Execution::ExecutionSpaceImpl::ExecutionSpaceRef<execution_space>() noexcept>
    >::
        template any_sender<
            stdexec::get_completion_scheduler<stdexec::set_value_t>.signature<Kokkos::Execution::ExecutionSpaceImpl::Scheduler<execution_space>() noexcept>,
            stdexec::get_completion_domain<stdexec::set_value_t>.signature<Kokkos::Execution::ExecutionSpaceImpl::Domain() noexcept>
        >;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    any_sender_t chain = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data) | THEN_INCREMENT(data);

    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, decltype(chain)>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);
    static_assert(std::same_as<
                  stdexec::__completion_scheduler_of_t<stdexec::set_value_t, decltype(chain)>,
                  typename AnySenderTest::scheduler_t
    >);

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
        recorder_listener_t::record(
            [extended_chain = std::move(extended_chain)]() mutable { stdexec::sync_wait(std::move(extended_chain)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

} // namespace Tests::ExecutionSpaceImpl
