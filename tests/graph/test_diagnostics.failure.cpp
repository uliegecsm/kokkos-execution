#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/graph.hpp"

#include "tests/utils/functors/no_op.hpp"

//! @test Check that @ref Kokkos::Execution::Impl::mandate_or_explain works properly.
int main() {
    const Kokkos::ScopeGuard guard{};

    const TEST_EXECUTION_SPACE exec{};

    const Kokkos::Execution::GraphContext<TEST_EXECUTION_SPACE> gctx{exec};
    exec::single_thread_context stc{};

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    stdexec::sender auto sndr = stdexec::when_all( // NOLINT(misc-const-correctness)
        stdexec::schedule(stc.get_scheduler()) | stdexec::continues_on(gctx.get_scheduler())
            | stdexec::then(functor_t{}),
        stdexec::schedule(gctx.get_scheduler()) | stdexec::then(functor_t{}));

    using op_state_t = stdexec::connect_result_t<decltype(sndr), rcvr_t>;

    //! This line will not compile and should properly explain why.
    static_assert(Kokkos::Execution::Impl::mandate_or_explain<
                  Kokkos::Execution::GraphImpl::RemainsOnGraphFor<op_state_t, TEST_EXECUTION_SPACE>
    >());
}
