#include "kokkos-execution/graph.hpp"

/**
 * @test A @c stdexec::when_all with two branches on @ref Kokkos::Execution::GraphContext,
 *       followed by work without any explicit scheduler provided.
 *
 * There is no completion scheduler in the environment after @c stdexec::when_all, yet the completion
 * domain is @ref Kokkos::Execution::GraphImpl::Domain, such that our customization fails.
 *
 * @verbatim
 * schedule(gctx) | then -- \
 *                           when_all --> then
 * schedule(gctx) | then -- /
 * @endverbatim
 */
int main() {
    const Kokkos::ScopeGuard guard{};

    const TEST_EXECUTION_SPACE exec{};

    const Kokkos::Execution::GraphContext<TEST_EXECUTION_SPACE> gctx{exec};

    stdexec::sender auto when_all = stdexec::when_all(
        stdexec::schedule(gctx.get_scheduler()) | stdexec::then([]() { }),
        stdexec::schedule(gctx.get_scheduler()) | stdexec::then([]() { }));

    //! Completion domain is @ref Kokkos::Execution::GraphImpl::Domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, decltype(when_all)>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    //! There is no completion scheduler.
    static_assert(!std::invocable<
                  stdexec::get_completion_scheduler_t<stdexec::set_value_t>,
                  stdexec::env_of_t<decltype(when_all)>
    >);

    stdexec::sync_wait(std::move(when_all) | stdexec::then([]() { })); // NOLINT(performance-move-const-arg)
}
