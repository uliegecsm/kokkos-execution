#ifndef KOKKOS_EXECUTION_EXAMPLES_CG_CG_HPP
#define KOKKOS_EXECUTION_EXAMPLES_CG_CG_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/fork_join.hpp"
#include "exec/repeat_until.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#include "linalg.hpp"

namespace Examples::CG {

template <typename MatrixType, typename RhsViewType, typename SolViewType, typename MagnitudeType>
stdexec::sender auto
    cg(stdexec::scheduler auto schd,
       MatrixType mat,  // NOLINT(performance-unnecessary-value-param)
       RhsViewType rhs, // NOLINT(performance-unnecessary-value-param)
       SolViewType sol, // NOLINT(performance-unnecessary-value-param)
       MagnitudeType tol,
       size_t max_iter = Kokkos::finite_max_v<size_t>) {
    using scalar_t = typename RhsViewType::non_const_value_type;

    const typename RhsViewType::non_const_type res(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "res"), rhs.size());
    const Kokkos::View<scalar_t, Kokkos::SharedHostPinnedSpace> res_res(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "res dot old"));
    const typename RhsViewType::non_const_type dir(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "dir"), rhs.size());
    const typename RhsViewType::non_const_type mat_dir(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "mat * dir"), rhs.size());
    const Kokkos::View<scalar_t[5], Kokkos::SharedHostPinnedSpace> scalars(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "scalars"));
    const Kokkos::View<size_t, Kokkos::HostSpace> iter_count(Kokkos::view_alloc("iter count"));

    const auto dir_mat_dir = Kokkos::subview(scalars, 0);
    const auto alpha = Kokkos::subview(scalars, 1);
    const auto alpha_neg = Kokkos::subview(scalars, 2);
    const auto res_res_new = Kokkos::subview(scalars, 3);
    const auto beta = Kokkos::subview(scalars, 4);

    // clang-format off
    return stdexec::read_env(stdexec::get_start_scheduler)
        | stdexec::let_value([=](auto hschd) {
            return stdexec::schedule(schd)
                | deep_copy(res, rhs)
                | spmv(scalar_t{-1}, mat, sol, scalar_t{1}, res) // NOLINT(performance-unnecessary-value-param)
                | dot(res_res, res, res)
                | deep_copy(dir, res)
                | stdexec::let_value([=]() {
                    return experimental::execution::repeat_until(
                        stdexec::schedule(schd)
                            | spmv(scalar_t{1}, mat, dir, scalar_t{0}, mat_dir)
                            | dot(dir_mat_dir, dir, mat_dir)
                            | stdexec::then(KOKKOS_LAMBDA() {
                                alpha() = res_res() / dir_mat_dir();
                                alpha_neg() = -alpha();
                            })
                            | experimental::execution::fork_join(
                                stdexec::continues_on(schd)
                                    | axpby(alpha, dir, scalar_t{1}, sol),
                                stdexec::continues_on(schd)
                                    | axpby(alpha_neg, mat_dir, scalar_t{1}, res)
                                    | dot(res_res_new, res, res)
                                    | stdexec::then(KOKKOS_LAMBDA() {
                                        beta() = res_res_new() / res_res();
                                        res_res() = res_res_new();
                                    }))
                            | axpby(scalar_t{1}, res, beta, dir)
                            | stdexec::continues_on(hschd)
                            | stdexec::then([=] {
                                return Kokkos::sqrt(Kokkos::abs(res_res())) < tol || ++iter_count() >= max_iter;
                            }));
                });
        });
    // clang-format on
}

} // namespace Examples::CG

#endif // KOKKOS_EXECUTION_EXAMPLES_CG_CG_HPP
