#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/ensure_started.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/execution_space.hpp"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Use @c experimental::execution::ensure_started with @c Kokkos::Execution::ExecutionSpaceContext
 * -----------------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly works
 * with @c experimental::execution::ensure_started.
 *
 * The tests can be found in @ref tests/execution_space/test_ensure_started.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class EnsureStartedTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        BeginParallelForEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
};

//! @test Check that @ref Kokkos::Execution::ExecutionSpaceContext does its duty well when used with @c experimental::execution::ensure_started.
TEST_F(EnsureStartedTest, then) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    const auto recorder_listener = std::make_shared<recorder_listener_t>();

    Kokkos::utils::callbacks::Manager::register_listener(recorder_listener);
    {
        auto sndr = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data)
                  | experimental::execution::ensure_started();

        using sndr_t = decltype(sndr);

        ASSERT_THAT(recorder_listener->recorded_events, ::testing::SizeIs(2));
        ASSERT_THAT(data(), ::testing::Eq(1));
        ASSERT_THAT(
            recorder_listener->recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch"))));

        /// The sender must be moved, *i.e.* consumed once, see
        /// https://github.com/NVIDIA/stdexec/blob/3680678fa0f1a5f6ebb300df45ae547b8bd8eef1/include/exec/detail/shared.hpp#L507-L510.
        static_assert(!stdexec::__decay_copyable<sndr_t&>);

        stdexec::sync_wait(std::move(sndr));

        ASSERT_THAT(recorder_listener->recorded_events, ::testing::SizeIs(2));

        //! The chain environment advertises the default domain, and completes on the default domain.
        static_assert(std::same_as<stdexec::__domain_of_t<stdexec::env_of_t<sndr_t>>, stdexec::default_domain>);
        static_assert(std::same_as<
                      stdexec::__detail::__completing_domain_t<stdexec::set_value_t, sndr_t, stdexec::env<>>,
                      stdexec::default_domain
        >);

        //! It has no completion scheduler for the value channel.
        static_assert(!Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t, stdexec::env<>>);
    }
    Kokkos::utils::callbacks::Manager::unregister_listener(recorder_listener.get());
}

} // namespace Tests::ExecutionSpaceImpl
