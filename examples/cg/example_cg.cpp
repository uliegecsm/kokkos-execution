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

        auto chain = cg(schd, std::move(mat), std::move(rhs), sol);

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
