#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/just_stopped.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c stdexec::then by @c Kokkos::Execution::ExecutionSpaceContext
 * --------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c stdexec::then.
 *
 * The tests can be found in @ref tests/execution_space/test_then.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class ThenTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        BeginParallelForEvent,
        AllocateDataEvent,
        DeallocateDataEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
    using variant_t = typename recorder_listener_t::event_variant_t;
};

//! @test Check traits of sender returned by @c stdexec::then when customized for @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename ThenTest::schedule_sender_t;

    //! Then sender.
    using functor_t = Tests::Utils::Functors::Increment<ThenTest::view_s_t, true>;
    using then_sndr_t = stdexec::transform_sender_result_t<
        decltype(stdexec::then(std::declval<schd_sndr_t>(), std::declval<functor_t>())),
        stdexec::env<>
    >;

    //! Models the execution space completing sender concept.
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::execution_space_completing_sender<then_sndr_t>);
    static_assert(std::same_as<Kokkos::Execution::Impl::exec_of_t<then_sndr_t>, TEST_EXECUTION_SPACE>);

    //! Does not model the dispatching sender concept.
    static_assert(!Kokkos::Execution::Impl::dispatching_sender<then_sndr_t>);

    //! The policy used in the parallel for created by the @c then sender has the expected launch bounds.
    using policy_t = typename then_sndr_t::closure_t::policy_t;
    static_assert(std::same_as<typename policy_t::launch_bounds, Kokkos::LaunchBounds<1>>);

    return true;
}
static_assert(test_sndr_traits());

//! @test Our customization is not selected. No value channel is added, such that it is not sync-waitable.
static_assert(Tests::Utils::check_continues_on_after_just_stopped<
              typename ThenTest::scheduler_t,
              stdexec::then_t,
              Tests::Utils::Functors::NoOp<false, false, false>
>());

/**
 * @test Check that @ref Kokkos::Execution::ExecutionSpaceContext does its duty well when used with @c stdexec::then
 *       within a chain started with @c stdexec::schedule.
 */
TEST_F(ThenTest, then_schedule) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data) | THEN_INCREMENT(data);

    using chain_t = decltype(chain);

    //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain domain.
    static_assert(std::same_as<stdexec::__domain_of_t<stdexec::env_of_t<chain_t>>, stdexec::default_domain>);
    static_assert(std::same_as<
                  stdexec::__detail::__completing_domain_t<stdexec::set_value_t, chain_t>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
                  decltype(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(chain))),
                  scheduler_t
    >);

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 2);
}

/**
 * @test Similar to @ref Tests::ExecutionSpaceImpl::ThenTest_then_schedule_Test, but the chain is scheduled
 *       with a @c starts_on.
 *
 * @todo Too many synchronizations.
 */
TEST_F(ThenTest, then_starts_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    //! Create a chain that does not start with a schedule sender.
    auto chain = stdexec::just() | THEN_INCREMENT(data) | THEN_INCREMENT(data);

    /// The chain cannot be queried for a completion scheduler.
    /// It may complete on the value channel or the error channel, since the @c stdexec::then functor is not @c noexcept.
    /// It hasn't been connected yet, so the domain is indeterminate.
    using chain_t = decltype(chain);

    static_assert(!Tests::Utils::has_completion_scheduler_for<chain_t, stdexec::set_value_t>);
    static_assert(Tests::Utils::has_completion_signatures<
                  chain_t,
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>
    >);

    static_assert(
        std::same_as<stdexec::__completion_domain_of_t<stdexec::set_value_t, chain_t>, stdexec::indeterminate_domain<>>);

    //! Call @c starts_on.
    auto starts_on = stdexec::starts_on(esc.get_scheduler(), std::move(chain));

    using starts_on_t = decltype(starts_on);

    //! It has a completion scheduler for the value channel.
    static_assert(Tests::Utils::has_completion_scheduler_for<starts_on_t, stdexec::set_value_t>);
    static_assert(std::same_as<
                  decltype(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(starts_on))),
                  scheduler_t
    >);

    static_assert(std::same_as<
                  std::invoke_result_t<stdexec::get_completion_signatures_t, starts_on_t, stdexec::env<>>,
                  stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(starts_on));

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        }
    }());

    ASSERT_EQ(data(), 2);
}

/**
 * @test If an exception is thrown while dispatching a @c Kokkos parallel region, it is properly caught
 *       and propagated in the error channel.
 *
 * Any parallel region successfully launched before the one that fails completes correctly because it will be
 * synchronized. Any sender downstream will not launch its parallel region.
 */
TEST_F(ThenTest, error_propagates) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    stdexec::sender auto sndr = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data)
                              | stdexec::then(Tests::Utils::Functors::ThrowsWhenCopied{})
                              | stdexec::then(KOKKOS_LAMBDA() {
                                    Kokkos::abort("The value channel should be used at this point.");
                                });

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable {
            ASSERT_THAT(
                Tests::Utils::Functors::MutableMoveToSyncWait{.sndr = std::move(sndr)},
                testing::ThrowsMessage<std::runtime_error>(
                    testing::HasSubstr("ThrowsWhenCopied: Throwing in copy constructor!")));
        }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")), // Increment: succeeds
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")), // ThrowsWhenCopied: throws on copy
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check the sequence of events when a @c Kokkos::View is captured in a chain of work.
 *
 * This test ensures that the deallocation happens after the kernels are finished.
 */
TEST_F(ThenTest, then_lifetime) {
    //! The context must be kept alive until the chain has completed.
    const context_t esc{exec};

    //! Create the chain in a scope.
    auto create_chain_in_scope = [&]() {
        const view_s_t data(Kokkos::view_alloc("data - shared space", exec));

        return stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data) | THEN_INCREMENT(data);
    };

    //! Run the whole test in a lambda.
    auto run_test = [&]() {
        auto chain = create_chain_in_scope();

        using chain_t = decltype(chain);

        //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain domain.
        static_assert(std::same_as<stdexec::__domain_of_t<stdexec::env_of_t<chain_t>>, stdexec::default_domain>);
        static_assert(std::same_as<
                      stdexec::__detail::__completing_domain_t<stdexec::set_value_t, chain_t>,
                      Kokkos::Execution::ExecutionSpaceImpl::Domain
        >);

        //! It has a completion scheduler for the value channel.
        static_assert(std::same_as<
                      decltype(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(chain))),
                      scheduler_t
        >);

        stdexec::sync_wait(std::move(chain));
    };

    ASSERT_THAT(
        recorder_listener_t::record(run_test),
        ContainsInOrder<variant_t>(
            Kokkos::utils::callbacks::AAllocateDataEvent(
                testing::Field(
                    &Kokkos::utils::callbacks::AllocateDataEvent::alloc,
                    testing::Field(
                        &Kokkos::utils::callbacks::AllocDescriptor::name, testing::StrEq("data - shared space")))),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")),
            Kokkos::utils::callbacks::ADeallocateDataEvent(
                testing::Field(
                    &Kokkos::utils::callbacks::DeallocateDataEvent::alloc,
                    testing::Field(
                        &Kokkos::utils::callbacks::AllocDescriptor::name, testing::StrEq("data - shared space"))))));
}

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_nothrow_transformable() {
    using sndr_then_t =
        decltype(stdexec::schedule(std::declval<typename ThenTest::scheduler_t>()) | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{}));

    static_assert(std::same_as<
                  stdexec::__demangle_t<sndr_then_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::then_t,
                      Tests::Utils::Functors::NoOp<false, false, false>,
                      typename ThenTest::schedule_sender_t
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::set_value_t,
                  sndr_then_t&&,
                  stdexec::env<>
    >);

    using sndr_then_maythrow_on_move_t =
        decltype(stdexec::schedule(std::declval<typename ThenTest::scheduler_t>()) | stdexec::then(Tests::Utils::Functors::NoOp<false, false, true>{}));

    static_assert(!stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::set_value_t,
                  sndr_then_maythrow_on_move_t&&,
                  stdexec::env<>
    >);

    return true;
}
static_assert(test_sndr_nothrow_transformable());

} // namespace Tests::ExecutionSpaceImpl
