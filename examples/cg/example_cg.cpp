#include "Kokkos_Core.hpp"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "fem_laplacian_1d.hpp"
#include "linalg.hpp"

namespace Examples::CG {

struct FEMLaplacian1DProblem {
   public:
    static bool run(const EXAMPLE_EXECUTION_SPACE& exec) {
        const Kokkos::utils::tests::scoped::callbacks::Manager kokkos_api_tracing_guard{};

        //! Problem setup.
        using scalar_t = double;
        using ordinal_t = int;
        using memory_space = typename EXAMPLE_EXECUTION_SPACE::memory_space;

        constexpr size_t size = 10;
        auto [mat, rhs, sol] = FEMLaplacian1D<scalar_t, ordinal_t, memory_space>::create(exec, size);

        //! Construct the CG solver.
        const Kokkos::Execution::ExecutionSpaceContext esc{exec};
        auto chain = cg(esc.get_scheduler(), std::move(mat), std::move(rhs), sol, /*residual tol=*/1e-14);

        //! Solve, with Kokkos API tracing.
        using recorder_listener_t = Kokkos::utils::callbacks::RecorderListener<
            Kokkos::utils::callbacks::BeginFenceEvent,
            Kokkos::utils::callbacks::BeginParallelForEvent,
            Kokkos::Execution::Impl::RecordEvent,
            Kokkos::Execution::Impl::WaitEvent
        >;
        const auto recorded_events = recorder_listener_t::record([&]() { stdexec::sync_wait(std::move(chain)); });

        /// Show the Kokkos API trace.
        for (const auto& recorded_event: recorded_events) {
            std::visit([](const auto& arg) { std::cout << "- " << arg << std::endl; }, recorded_event);
        }

        //! Check the solution.
        return FEMLaplacian1D<scalar_t, ordinal_t, memory_space>::check(exec, sol, /*error tol=*/1e-14);
    }
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
    {
        success &= Examples::CG::FEMLaplacian1DProblem::run(EXAMPLE_EXECUTION_SPACE{});
    }

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
