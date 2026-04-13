#include "kokkos-execution/execution_space.hpp"

/**
 * @test A @c stdexec::when_all with a single branch on @ref Kokkos::Execution::ExecutionSpaceContext,
 *       followed by work without any explicit scheduler provided.
 *
 * It is similar to @ref Tests::ExecutionSpaceImpl::WhenAllTest_single_branch_followed_by_self_Test.
 * However, there is no completion scheduler in the environment after @c stdexec::when_all, yet the completion
 * domain is @ref Kokkos::Execution::ExecutionSpaceImpl::Domain, such that our customization fails.
 *
 * @verbatim
 * schedule(esc) | then -- when_all --> then
 * @endverbatim
 */
int main() {
    const Kokkos::ScopeGuard guard{};

    const TEST_EXECUTION_SPACE exec{};

    const Kokkos::Execution::ExecutionSpaceContext<TEST_EXECUTION_SPACE> ctx{exec};

    stdexec::sender auto when_all = stdexec::when_all(stdexec::schedule(ctx.get_scheduler()) | stdexec::then([]() { }));

    //! Completion domain is @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, decltype(when_all)>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! There is no completion scheduler.
    static_assert(!std::invocable<
                  stdexec::get_completion_scheduler_t<stdexec::set_value_t>,
                  stdexec::env_of_t<decltype(when_all)>
    >);

    stdexec::sync_wait(std::move(when_all) | stdexec::then([]() { })); // NOLINT(performance-move-const-arg)
}
