#ifndef KOKKOS_EXECUTION_EXAMPLES_CG_FEM_LAPLACIAN_1D_HPP
#define KOKKOS_EXECUTION_EXAMPLES_CG_FEM_LAPLACIAN_1D_HPP

#include "Kokkos_Core.hpp"

#include "linalg.hpp"

namespace Examples::CG {

/**
 * @verbatim
 * 1  0                     0
 * 0  2 -1                  0
 *       *                  *
 *           *              *
 *              *           *
 *             -1  2 -1     0
 *                -1  1     2 / size
 * @endverbatim
 */
template <typename ScalarType, Kokkos::MemorySpace Mem>
struct FEMLaplacian1D {
    using scalar_t = ScalarType;
    using ordinal_t = typename Mem::size_type;

    using matrix_t = CrsMatrix<scalar_t, ordinal_t, Mem>;
    using vector_t = Kokkos::View<scalar_t*, Mem>;

    template <Kokkos::ExecutionSpace Exec>
    static FEMLaplacian1D create(const Exec& exec, const size_t size) {
        const auto nnz = 3 * size - 2;

        using row_map_t = typename matrix_t::graph_t::row_map_t::non_const_type;
        using entries_t = typename matrix_t::graph_t::entries_t;
        using values_t = typename matrix_t::values_t;

        row_map_t row_map( // NOLINT(misc-const-correctness)
            Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row map"),
            size + 1);
        entries_t entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "entries"), nnz);
        values_t values(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "values"), nnz);
        vector_t rhs_(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rhs"), size);
        vector_t guess_(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "guess"), size);

        Kokkos::parallel_for(
            "FEMLaplacian1D", Kokkos::RangePolicy(exec, 0, size), KOKKOS_LAMBDA(const ordinal_t irow) {
                if (irow == 0) {
                    row_map(0) = 0;
                    row_map(1) = 2;

                    entries(0) = 0;
                    entries(1) = 1;

                    values(0) = 1;
                    values(1) = 0;

                    rhs_(0) = 0.;
                } else if (irow < size - 1) {
                    const auto offset = 3 * (irow - 1) + 2;

                    row_map(irow + 1) = offset + 3;

                    entries(offset + 0) = irow - 1;
                    entries(offset + 1) = irow + 0;
                    entries(offset + 2) = irow + 1;

                    values(offset + 0) = irow == 1 ? 0 : -1;
                    values(offset + 1) = 2;
                    values(offset + 2) = -1;

                    rhs_(irow) = 0.;
                } else {
                    const auto offset = 3 * (irow - 1) + 2;

                    row_map(irow + 1) = offset + 2;

                    entries(offset + 0) = irow - 1;
                    entries(offset + 1) = irow + 0;

                    values(offset + 0) = -1;
                    values(offset + 1) = 1;

                    rhs_(irow) = static_cast<scalar_t>(2) / size;
                }

                guess_(irow) = rhs_(irow);
            });

        return {
            .matrix =
                {.graph = {.row_map = std::move(row_map), .entries = std::move(entries)}, .values = std::move(values)},
            .rhs = std::move(rhs_),
            .guess = std::move(guess_)
        };
    }

    matrix_t matrix;
    vector_t rhs;
    vector_t guess;
};

} // namespace Examples::CG

#endif // KOKKOS_EXECUTION_EXAMPLES_CG_FEM_LAPLACIAN_1D_HPP
