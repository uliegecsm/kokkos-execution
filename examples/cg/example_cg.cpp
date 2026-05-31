#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "fem_laplacian_1d.hpp"
#include "linalg.hpp"

namespace Examples::CG {

class FEMLaplacian1DProblem {
   public:
    static void run(const EXAMPLE_EXECUTION_SPACE& exec) {
        constexpr size_t size = 10;

        using scalar_t = double;
        using memory_space = typename EXAMPLE_EXECUTION_SPACE::memory_space;

        auto [mat, rhs, sol] = FEMLaplacian1D<scalar_t, memory_space>::create(exec, size);

        const Kokkos::Execution::ExecutionSpaceContext esc{exec};
        const auto schd = esc.get_scheduler();

        experimental::execution::single_thread_context stc{};
        const auto host_schd = stc.get_scheduler();

        auto chain = cg(schd, host_schd, std::move(mat), std::move(rhs), sol, 1e-14);

        stdexec::sync_wait(std::move(chain));
    }
};

} // namespace Examples::CG

int main(int argc, char* argv[]) {
    const Kokkos::ScopeGuard guard{argc, argv};
    {
        Examples::CG::FEMLaplacian1DProblem::run(EXAMPLE_EXECUTION_SPACE{});
    }

    return EXIT_SUCCESS;
}
