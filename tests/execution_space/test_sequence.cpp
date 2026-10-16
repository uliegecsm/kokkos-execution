#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/sequence.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_rcvr_env_queryable_with.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Interaction of @c exec::sequence with @c Kokkos::Execution::ExecutionSpaceContext
 * ---------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly interacts with
 * @c exec::sequence.
 *
 * The tests can be found in @ref tests/execution_space/test_sequence.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(SequenceTest)
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
    using variant_t = typename recorder_listener_t::event_variant_t;
};

//! @test Check traits of a @c exec::sequence sender.
consteval bool test_sequence_sndr_traits() {
    using schd_sndr_t = typename TEST_CATEGORY(SequenceTest)::schedule_sender_t;
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    using seq_sndr_t =
        decltype(exec::sequence(std::declval<schd_sndr_t>(), stdexec::just() | stdexec::then(functor_t{})));

    //! It is a dependent sender.
    static_assert(stdexec::dependent_sender<seq_sndr_t>);

    //! Has the expected completion signatures.
    static_assert(
        stdexec::get_completion_signatures<seq_sndr_t, stdexec::env<>>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, seq_sndr_t, stdexec::env<>>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! Has the expected completion scheduler.
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, seq_sndr_t, stdexec::env<>>,
                  Kokkos::Execution::ExecutionSpaceImpl::Scheduler<TEST_EXECUTION_SPACE>
    >);

    //! Is connectable.
    static_assert(stdexec::sender_to<seq_sndr_t, Tests::Utils::SinkReceiver>);

    static_assert(std::same_as<
                  stdexec::transform_sender_result_t<seq_sndr_t, stdexec::env_of_t<Tests::Utils::SinkReceiver>>,
                  stdexec::__seq::__sndr<schd_sndr_t, stdexec::then_t::__sender<functor_t, stdexec::just_t::__sender<>>>
    >);

    //! It is nothrow connectable.
    static_assert(stdexec::__nothrow_connectable<seq_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}

static_assert(test_sequence_sndr_traits());

/**
 * @test Check the behavior of a @c exec::sequence of @c stdexec::then scheduled on @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @todo There are too many synchronizations.
 */
TEST_F(TEST_CATEGORY(SequenceTest), many_then) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<int, on_device>;

    auto sndr = exec::sequence(
                    stdexec::schedule(esc.get_scheduler()),
                    stdexec::just() | stdexec::then(functor_t{.prev = 0, .value = 1, .data = data.data()}),
                    stdexec::just() | stdexec::then(functor_t{.prev = 1, .value = 2, .data = data.data()}),
                    stdexec::just() | stdexec::then(functor_t{.prev = 3, .value = 3, .data = data.data()}))
              | stdexec::then(functor_t{.prev = 6, .value = 4, .data = data.data()});

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(4)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(7)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        }
    }());

    ASSERT_EQ(data(), 10);
}

/**
 * @test Check the behavior of a @c exec::sequence of @c stdexec::then scheduled on @ref Kokkos::Execution::ExecutionSpaceContext
 *       and then transferred to a @c exec::static_thread_pool.
 */
TEST_F(TEST_CATEGORY(SequenceTest), hop_on_hop_off) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};
    exec::static_thread_pool pool{1};

    using functor_d_t = Tests::Utils::Functors::LoadCheckAdd<int, on_device>;
    using functor_h_t = Tests::Utils::Functors::LoadCheckAdd<int, false>;

    auto sndr = exec::sequence(
                    stdexec::schedule(pool.get_scheduler()),
                    stdexec::just() | stdexec::then(functor_h_t{.prev = 0, .value = 1, .data = data.data()}),
                    stdexec::schedule(esc.get_scheduler()),
                    stdexec::just() | stdexec::then(functor_d_t{.prev = 1, .value = 1, .data = data.data()}),
                    stdexec::just() | stdexec::then(functor_d_t{.prev = 2, .value = 1, .data = data.data()}),
                    stdexec::schedule(pool.get_scheduler()),
                    stdexec::just() | stdexec::then(functor_h_t{.prev = 3, .value = 1, .data = data.data()}))
              | stdexec::then(functor_h_t{.prev = 4, .value = 1, .data = data.data()});

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, decltype(sndr), stdexec::env<>>,
                  experimental::execution::_pool_::_static_thread_pool::domain
    >);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(4)));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")));
        }
    }());

    ASSERT_EQ(data(), 5);
}

} // namespace Tests::ExecutionSpaceImpl
