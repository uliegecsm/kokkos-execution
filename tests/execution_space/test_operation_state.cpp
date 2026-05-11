#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/labeled.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Operation state @c Kokkos::Execution::ExecutionSpaceImpl::OpState
 * -----------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::ExecutionSpaceImpl::OpState
 * and its related types.
 *
 * The tests can be found in @ref tests/execution_space/test_operation_state.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

class OpStateTest : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE> { };

//! @test Check traits of @c Kokkos::Execution::ExecutionSpaceImpl::OpState.
consteval bool test_op_state_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename OpStateTest::schedule_sender_t;

    //! Parallel for closure.
    using functor_t = Tests::Utils::Functors::SumIndices<typename OpStateTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using clsr_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<std::string, functor_t, policy_t>;

    //! Receiver.
    using rcvr_t = Tests::Utils::SinkReceiver;

    //! Operation state.
    using op_state_t = Kokkos::Execution::ExecutionSpaceImpl::OpState<schd_sndr_t, rcvr_t, clsr_t>;

    //! Models the @ref Tests::Utils::operation_state concept.
    static_assert(Tests::Utils::operation_state<op_state_t>);

    //! By inheriting from @ref Kokkos::Execution::Impl::Immovable, it is neither moveable nor copyable.
    static_assert(std::derived_from<op_state_t, Kokkos::Execution::Impl::Immovable>);
    static_assert(!std::move_constructible<op_state_t>);
    static_assert(!std::is_move_assignable_v<op_state_t>);

    static_assert(!std::copy_constructible<op_state_t>);
    static_assert(!std::is_copy_assignable_v<op_state_t>);

    return true;
}
static_assert(test_op_state_traits());

/**
 * @test Check construction of operation state from a parallel for sender
 *       when passing the sender as a @c const reference.
 */
constexpr bool test_op_state_passed_by_const_ref() {
    using sndr_t = decltype(Kokkos::Execution::parallel_for(
        stdexec::schedule(std::declval<typename OpStateTest::context_t>().get_scheduler()),
        "hello from pfor",
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 10),
        Tests::Utils::Functors::Labeled<'a'>{}));

    static_assert(!std::is_const_v<sndr_t>);

    //! Connect the sender as a @c const reference.
    using op_state_from_sndr_const_ref_t = stdexec::connect_result_t<const sndr_t&, Tests::Utils::SinkReceiver>;

    static_assert(std::same_as<
                  op_state_from_sndr_const_ref_t,
                  Kokkos::Execution::ExecutionSpaceImpl::OpState<
                      const typename OpStateTest::schedule_sender_t&,
                      Tests::Utils::SinkReceiver,
                      Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
                          std::string,
                          Tests::Utils::Functors::Labeled<'a'>,
                          Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
                      >
                  >
    >);

    return true;
}
static_assert(test_op_state_passed_by_const_ref());

/**
 * @test Check that @ref Kokkos::Execution::ExecutionSpaceImpl::OpStateBase possibly uses
 *       an @ref Kokkos::Execution::Impl::Event only if
 *       the receiver environment is queryable for a delegation scheduler and the execution space supports events.
 */
template <stdexec::receiver Rcvr>
consteval bool test_delegate_completion_with_event() {
    using closure_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string_view,
        Tests::Utils::Functors::Labeled<'a'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;
    using opstate_t =
        Kokkos::Execution::ExecutionSpaceImpl::OpState<typename OpStateTest::schedule_sender_t, Rcvr, closure_t>;
    using opstate_base_t = Kokkos::Execution::ExecutionSpaceImpl::OpStateBase<Rcvr, closure_t>;
    using may_delegate_completion_with_event_t =
        Kokkos::Execution::ExecutionSpaceImpl::MayDelegateCompletionWithEvent<Rcvr, TEST_EXECUTION_SPACE>;

    static_assert(std::derived_from<opstate_t, opstate_base_t>);
    static_assert(std::derived_from<opstate_base_t, may_delegate_completion_with_event_t>);

    //! When delegating, additional space is taken by the storage of the operation state. Otherwise, it only stores the receiver.
    if constexpr (Kokkos::Execution::ExecutionSpaceImpl::delegate_completion_with_event<Rcvr, TEST_EXECUTION_SPACE>) {
        static_assert(sizeof(may_delegate_completion_with_event_t) > sizeof(Rcvr));
        static_assert(std::same_as<
                      typename may_delegate_completion_with_event_t::receiver_t,
                      Kokkos::Execution::ExecutionSpaceImpl::WaitEventReceiver<Rcvr, TEST_EXECUTION_SPACE>
        >);
        static_assert(std::same_as<
                      typename may_delegate_completion_with_event_t::receiver_t::event_t,
                      Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE>
        >);
    } else {
        static_assert(sizeof(may_delegate_completion_with_event_t) == sizeof(Rcvr));
    }

    return true;
}
static_assert(test_delegate_completion_with_event<Tests::Utils::SinkReceiver>());
static_assert(test_delegate_completion_with_event<Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE>>());

//! @test Check construction, query for execution space instance, and start.
TEST_F(OpStateTest, construct_query_and_start) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "witness - shared space"));

    const context_t esc{exec};

    Kokkos::Execution::Impl::ParallelForData pfor_data{
        "hello from pfor", Tests::Utils::Functors::SumIndices{.data = witness}, Kokkos::RangePolicy(exec, 2, size)};
    Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure clsr{.data = std::move(pfor_data)};

    auto op_state = Kokkos::Execution::ExecutionSpaceImpl::OpState{
        stdexec::schedule(esc.get_scheduler()), Tests::Utils::SinkReceiver{}, std::move(clsr)};

    ASSERT_EQ(Kokkos::Execution::Impl::get_exec(op_state).get(), exec);

    op_state.start();
    exec.fence();

    ASSERT_EQ(witness(), size / 2 * (size - 1) - 1);
}

//! @test Check construction of flattened operation state from two parallel for senders.
consteval bool test_op_state_flattened_from_two() {
    using sndr_t = decltype(Kokkos::Execution::parallel_for(
        Kokkos::Execution::parallel_for(
            stdexec::schedule(std::declval<typename OpStateTest::context_t>().get_scheduler()),
            "hello from pfor",
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 10),
            Tests::Utils::Functors::Labeled<'a'>{}),
        "hello again from pfor",
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 10),
        Tests::Utils::Functors::Labeled<'b'>{}));

    using op_state_t = stdexec::connect_result_t<sndr_t&&, Tests::Utils::SinkReceiver>;

    using clsr_0_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string,
        Tests::Utils::Functors::Labeled<'a'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;
    using clsr_1_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string,
        Tests::Utils::Functors::Labeled<'b'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;

    static_assert(std::same_as<
                  op_state_t,
                  Kokkos::Execution::ExecutionSpaceImpl::OpState<
                      typename OpStateTest::schedule_sender_t,
                      Tests::Utils::SinkReceiver,
                      clsr_0_t,
                      clsr_1_t
                  >
    >);

    static_assert(!std::is_nothrow_constructible_v<op_state_t, sndr_t, Tests::Utils::SinkReceiver, clsr_0_t, clsr_1_t>);

    static_assert(stdexec::__tuple_size_v<typename op_state_t::closures_t> == 2);

    return true;
}
static_assert(test_op_state_flattened_from_two());

//! @test Check construction of flattened operation state from three parallel for senders.
consteval bool test_op_state_flattened_from_three() {
    using sndr_t = decltype(Kokkos::Execution::parallel_for(
        Kokkos::Execution::parallel_for(
            Kokkos::Execution::parallel_for(
                stdexec::schedule(std::declval<typename OpStateTest::context_t>().get_scheduler()),
                "hello from pfor",
                Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 10),
                Tests::Utils::Functors::Labeled<'a'>{}),
            "hello again from pfor",
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 10),
            Tests::Utils::Functors::Labeled<'b'>{}),
        "hello one more time from pfor",
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 10),
        Tests::Utils::Functors::Labeled<'c'>{}));

    using op_state_t = stdexec::connect_result_t<sndr_t&&, Tests::Utils::SinkReceiver>;

    using clsr_0_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string,
        Tests::Utils::Functors::Labeled<'a'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;
    using clsr_1_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string,
        Tests::Utils::Functors::Labeled<'b'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;
    using clsr_2_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string,
        Tests::Utils::Functors::Labeled<'c'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;

    static_assert(std::same_as<
                  op_state_t,
                  Kokkos::Execution::ExecutionSpaceImpl::OpState<
                      typename OpStateTest::schedule_sender_t,
                      Tests::Utils::SinkReceiver,
                      clsr_0_t,
                      clsr_1_t,
                      clsr_2_t
                  >
    >);

    static_assert(
        !std::is_nothrow_constructible_v<op_state_t, sndr_t, Tests::Utils::SinkReceiver, clsr_0_t, clsr_1_t, clsr_2_t>);

    static_assert(stdexec::__tuple_size_v<typename op_state_t::closures_t> == 3);

    return true;
}
static_assert(test_op_state_flattened_from_three());

//! @test Check construction of flattened operation state from three parallel for senders with mixed tags.
consteval bool test_op_state_flattened_from_three_mixed_tags() {
    using sndr_t = decltype(stdexec::then(
        stdexec::bulk(
            Kokkos::Execution::parallel_for(
                stdexec::schedule(std::declval<typename OpStateTest::context_t>().get_scheduler()),
                "hello from pfor",
                Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::IndexType<size_t>>(0, 10),
                Tests::Utils::Functors::Labeled<'a'>{}),
            stdexec::par,
            10,
            Tests::Utils::Functors::Labeled<'b'>{}),
        Tests::Utils::Functors::Labeled<'c'>{}));

    using op_state_t = stdexec::connect_result_t<sndr_t&&, Tests::Utils::SinkReceiver>;

    using clsr_0_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string,
        Tests::Utils::Functors::Labeled<'a'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::IndexType<size_t>>
    >;
    using clsr_1_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string_view,
        Tests::Utils::Functors::Labeled<'b'>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;
    using clsr_2_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
        std::string_view,
        Kokkos::Execution::ExecutionSpaceImpl::ThenWrapper<Tests::Utils::Functors::Labeled<'c'>>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::LaunchBounds<1>>
    >;

    static_assert(std::same_as<
                  op_state_t,
                  Kokkos::Execution::ExecutionSpaceImpl::OpState<
                      typename OpStateTest::schedule_sender_t,
                      Tests::Utils::SinkReceiver,
                      clsr_0_t,
                      clsr_1_t,
                      clsr_2_t
                  >
    >);

    static_assert(
        !std::is_nothrow_constructible_v<op_state_t, sndr_t, Tests::Utils::SinkReceiver, clsr_0_t, clsr_1_t, clsr_2_t>);

    static_assert(stdexec::__tuple_size_v<typename op_state_t::closures_t> == 3);

    return true;
}
static_assert(test_op_state_flattened_from_three_mixed_tags());

} // namespace Tests::ExecutionSpaceImpl
