#include "Kokkos_Core.hpp"
#include "Kokkos_Profiling_ProfileSection.hpp"

#include "kokkos-utils/callbacks/EventInProfileSectionMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegexMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"
#include "kokkos-execution/graph.hpp"

#include "fem_laplacian_1d.hpp"
#include "solver.hpp"

namespace Examples::CG {

template <Kokkos::ExecutionSpace Exec>
struct LinearProblem {
   public:
    using scalar_t = double;
    using magnitude_t = double;
    using ordinal_t = int;
    using memory_space = typename Exec::memory_space;

    using vector_t = Kokkos::View<scalar_t*, memory_space>;
    using matrix_t = CrsMatrix<scalar_t, ordinal_t, memory_space>;

   public:
    explicit LinearProblem(const Exec& exec, const size_t size = 10) {
        auto [mat_, rhs_, sol_] = FEMLaplacian1D<scalar_t, magnitude_t, ordinal_t, memory_space>::create(exec, size);
        this->mat = std::move(mat_);
        this->rhs = std::move(rhs_);
        this->sol = std::move(sol_);
    }

    auto solve(stdexec::scheduler auto schd) const {
        //! Define the CG solver.
        auto sndr = cg(schd, mat, rhs, sol, /*residual tol=*/1e-14);

        //! Execute the CG solver.
        Kokkos::Profiling::ProfilingSection profile_section("CG solver execution");
        profile_section.start();
        auto [summary] = stdexec::sync_wait(std::move(sndr)).value();
        profile_section.stop();

        return summary;
    }

    bool check(const Exec& exec) const {
        //! Check the solution.
        return FEMLaplacian1D<scalar_t, magnitude_t, ordinal_t, memory_space>::check(exec, sol, /*error tol=*/1e-14);
    }

   protected:
    matrix_t mat;
    vector_t rhs, sol;
};

} // namespace Examples::CG

int main(int argc, char* argv[]) {
    //! Disable @c Kokkos::Tools fences. Must be called before @c Kokkos::initialize.
    Kokkos::Tools::Experimental::set_request_tool_settings_callback(
        [](const uint32_t, Kokkos::Tools::Experimental::ToolSettings* settings) -> void {
            settings->requires_global_fencing = false;
        });

    bool success = true;

    const Kokkos::ScopeGuard guard{argc, argv};
    const Kokkos::utils::tests::scoped::callbacks::Manager kokkos_callbacks_guard{};
    {
        const EXAMPLE_EXECUTION_SPACE exec{};

        //! Setup the problem.
        const Examples::CG::LinearProblem<EXAMPLE_EXECUTION_SPACE> problem{exec};

        /// Solve the problem. Under Kokkos API tracing (code should be shortened).
        /// const auto [summary, events] = recorder_t::record(section_matcher_t("CG solver execution"),
        ///                                                   [&] { return problem.solve(); });
        using event_in_profile_section_recorder_t = Kokkos::utils::callbacks::RecorderListener<
            Kokkos::utils::callbacks::EventInProfileSectionMatcher<Kokkos::utils::callbacks::EventRegexMatcher>,
            Kokkos::Impl::type_list<
                Kokkos::utils::callbacks::BeginFenceEvent,
                Kokkos::utils::callbacks::BeginParallelForEvent,
                Kokkos::utils::callbacks::BeginParallelReduceEvent,
                Kokkos::utils::callbacks::CreateProfileSectionEvent,
                Kokkos::utils::callbacks::StartProfileSectionEvent,
                Kokkos::utils::callbacks::StopProfileSectionEvent,
                Kokkos::utils::callbacks::DestroyProfileSectionEvent,
                Kokkos::Execution::Impl::RecordEvent,
                Kokkos::Execution::Impl::WaitEvent,
                Kokkos::Execution::GraphImpl::GraphAddNodeEvent,
                Kokkos::Execution::GraphImpl::GraphCreateEvent,
                Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
                Kokkos::Execution::GraphImpl::GraphSubmitEvent
            >
        >;

        Kokkos::utils::callbacks::EventInProfileSectionMatcher matcher{
            .matcher = Kokkos::utils::callbacks::EventRegexMatcher{.regex = std::regex("CG solver execution")}};
        const auto recorder = std::make_shared<event_in_profile_section_recorder_t>(std::move(matcher));
        Kokkos::utils::callbacks::Manager::register_listener(recorder);

        //const Kokkos::Execution::ExecutionSpaceContext esc{exec};
        //const auto summary = problem.solve(esc.get_scheduler());
        const Kokkos::Execution::GraphContext gctx{exec};
        const auto summary = problem.solve(gctx.get_scheduler());

        //! Print the solver summary.
        std::cout << "CG converged in " << summary.iter_count
                  << " iterations, with residual norm squared = " << summary.res_norm_sq << std::endl;

        //! Show the Kokkos API trace.
        recorder->report(std::cout);

        //! Check the solution.
        success &= problem.check(exec);

        Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());
    }

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
