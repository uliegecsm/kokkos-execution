#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "gtest/gtest.h"

#include "kokkos-execution/graph.hpp"

#include "tests/utils/category.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/graph_context.hpp"

namespace Tests::GraphImpl {

class TEST_CATEGORY(RemainsOnGraphForTest)
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE> { };

//! @test Check @ref Kokkos::Execution::GraphImpl::connect_result_remains_on_graph_for for a non-dependent sender on @ref Kokkos::Execution::GraphImpl::Domain.
TEST_F(TEST_CATEGORY(RemainsOnGraphForTest), non_dependent_sender) {
    const context_t gctx{exec};
    exec::single_thread_context stc{};

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    auto sndr_A = stdexec::schedule(stc.get_scheduler()) //| stdexec::then(functor_t{})
                | stdexec::continues_on(gctx.get_scheduler()) | stdexec::then(functor_t{});

    auto sndr_B = stdexec::schedule(stc.get_scheduler()) //| stdexec::then(functor_t{})
                | stdexec::continues_on(gctx.get_scheduler()) | stdexec::then(functor_t{});

    stdexec::sender auto sndr = stdexec::when_all(sndr_A, sndr_B);

    using op_state_t = stdexec::connect_result_t<
        decltype(sndr),
        Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>
    >;

    using op_state_A = stdexec::connect_result_t<decltype(sndr_A), typename op_state_t::WhenAllChildReceiver>;
    using op_state_B = stdexec::connect_result_t<decltype(sndr_B), typename op_state_t::WhenAllChildReceiver>;

    static_assert(std::same_as<stdexec::__tuple<op_state_A, op_state_B>, typename op_state_t::children_op_states_t>);

    using result_t =
        decltype(Kokkos::Execution::GraphImpl::RemainsOnGraphFor<op_state_t, TEST_EXECUTION_SPACE>::diagnose());

    static_assert(
        std::same_as<
            result_t,
            Kokkos::Execution::Impl::diagnoses<
                stdexec::_ERROR_<
                    stdexec::_WHAT_(stdexec::_INVALID_ARGUMENT_),
                    stdexec::_WHY_(Kokkos::Execution::GraphImpl::ESCAPING_FROM_THE_GRAPH_IS_NOT_SUPPORTED),
                    Kokkos::Execution::Impl::WITH_OPSTATE_TYPE<typename op_state_A::inner_op_state_t::inner_op_state_t>,
                    Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<Kokkos::Serial>
                >,
                stdexec::_ERROR_<
                    stdexec::_WHAT_(stdexec::_INVALID_ARGUMENT_),
                    stdexec::_WHY_(Kokkos::Execution::GraphImpl::ESCAPING_FROM_THE_GRAPH_IS_NOT_SUPPORTED),
                    Kokkos::Execution::Impl::WITH_OPSTATE_TYPE<typename op_state_B::inner_op_state_t::inner_op_state_t>,
                    Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<Kokkos::Serial>
                >
            >
        >);


    // static_assert(Kokkos::Execution::Impl::diagnose<
    //               Kokkos::Execution::GraphImpl::RemainsOnGraphFor,
    //               op_state_t,
    //               TEST_EXECUTION_SPACE
    // >());

    // stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)
}

} // namespace Tests::GraphImpl

// Kokkos::Execution::Impl::diagnoses<stdexec::_ERROR_<stdexec::_WHAT_ (stdexec::_INVALID_ARGUMENT_), stdexec::_WHY_ (Kokkos::Execution::GraphImpl::ESCAPING_FROM_THE_GRAPH_IS_NOT_SUPPORTED), Kokkos::Execution::Impl::WITH_OPSTATE_TYPE<stdexec::__opstate<stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>, Kokkos::Execution::GraphImpl::ContinuesOnReceiver<Kokkos::Execution::GraphImpl::ContinuesOnOpState<Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>, Kokkos::Execution::Impl::Receiver<Kokkos::Execution::GraphImpl::OpState<stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>, Kokkos::Execution::GraphImpl::WhenAllOpState<Kokkos::Serial, Kokkos::Execution::Impl::SyncWait::Receiver<Kokkos::Serial, std::integral_constant<bool, true>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>>::WhenAllChildReceiver, Kokkos::Execution::GraphImpl::ThenClosure<Kokkos::Serial, Tests::Utils::Functors::NoOp<false, false, false>>>, stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>>>, stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>>>>, Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<Kokkos::Serial>>, stdexec::_ERROR_<stdexec::_WHAT_ (stdexec::_INVALID_ARGUMENT_), stdexec::_WHY_ (Kokkos::Execution::GraphImpl::ESCAPING_FROM_THE_GRAPH_IS_NOT_SUPPORTED), Kokkos::Execution::Impl::WITH_OPSTATE_TYPE<stdexec::__opstate<stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>, Kokkos::Execution::GraphImpl::ContinuesOnReceiver<Kokkos::Execution::GraphImpl::ContinuesOnOpState<Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>, Kokkos::Execution::Impl::Receiver<Kokkos::Execution::GraphImpl::OpState<stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>, Kokkos::Execution::GraphImpl::WhenAllOpState<Kokkos::Serial, Kokkos::Execution::Impl::SyncWait::Receiver<Kokkos::Serial, std::integral_constant<bool, true>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>>::WhenAllChildReceiver, Kokkos::Execution::GraphImpl::ThenClosure<Kokkos::Serial, Tests::Utils::Functors::NoOp<false, false, false>>>, stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>>>, stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>>>>, Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<Kokkos::Serial>>>
// Kokkos::Execution::Impl::diagnoses<stdexec::_ERROR_<stdexec::_WHAT_ (stdexec::_INVALID_ARGUMENT_), stdexec::_WHY_ (Kokkos::Execution::GraphImpl::ESCAPING_FROM_THE_GRAPH_IS_NOT_SUPPORTED), Kokkos::Execution::Impl::WITH_OPSTATE_TYPE<Kokkos::Execution::GraphImpl::ContinuesOnOpState<Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>, Kokkos::Execution::Impl::Receiver<Kokkos::Execution::GraphImpl::OpState<stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>, Kokkos::Execution::GraphImpl::WhenAllOpState<Kokkos::Serial, Kokkos::Execution::Impl::SyncWait::Receiver<Kokkos::Serial, std::integral_constant<bool, true>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>>::WhenAllChildReceiver, Kokkos::Execution::GraphImpl::ThenClosure<Kokkos::Serial, Tests::Utils::Functors::NoOp<false, false, false>>>, stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>>>>, Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<Kokkos::Serial>>, stdexec::_ERROR_<stdexec::_WHAT_ (stdexec::_INVALID_ARGUMENT_), stdexec::_WHY_ (Kokkos::Execution::GraphImpl::ESCAPING_FROM_THE_GRAPH_IS_NOT_SUPPORTED), Kokkos::Execution::Impl::WITH_OPSTATE_TYPE<Kokkos::Execution::GraphImpl::ContinuesOnOpState<Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>, Kokkos::Execution::Impl::Receiver<Kokkos::Execution::GraphImpl::OpState<stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>, Kokkos::Execution::GraphImpl::WhenAllOpState<Kokkos::Serial, Kokkos::Execution::Impl::SyncWait::Receiver<Kokkos::Serial, std::integral_constant<bool, true>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>, stdexec::then_t::__sender<Tests::Utils::Functors::NoOp<false, false, false>, stdexec::__sexpr<stdexec::continues_on_t, Kokkos::Execution::GraphImpl::Scheduler<Kokkos::Serial>, stdexec::__sexpr<stdexec::schedule_from_t, stdexec::__, stdexec::__run_loop::__basic_run_loop<stdexec::run_loop, stdexec::env<>>::__scheduler::__sndr_t>>>>::WhenAllChildReceiver, Kokkos::Execution::GraphImpl::ThenClosure<Kokkos::Serial, Tests::Utils::Functors::NoOp<false, false, false>>>, stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>>>>, Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<Kokkos::Serial>>>
