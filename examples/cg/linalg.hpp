#ifndef KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
#define KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/fork_join.hpp"
#include "exec/repeat_until.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Examples::CG {

template <typename OrdinalType, Kokkos::MemorySpace Mem>
struct CrsGraph {
    using ordinal_t = OrdinalType;

    using row_map_t = Kokkos::View<const ordinal_t*, Mem>;
    using entries_t = Kokkos::View<ordinal_t*, Mem>;

    row_map_t row_map;
    entries_t entries;
};

template <typename ScalarType, typename OrdinalType, Kokkos::MemorySpace Mem>
struct CrsMatrix {
    using scalar_t = ScalarType;
    using ordinal_t = OrdinalType;

    using graph_t = CrsGraph<ordinal_t, Mem>;
    using values_t = Kokkos::View<scalar_t*, Mem>;

    using const_t = CrsMatrix<const scalar_t, ordinal_t, Mem>;

    graph_t graph;
    values_t values;
};

template <typename DstViewType, typename SrcViewType>
struct DeepCopy {
    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T idx) const {
        dst(idx) = src(idx);
    }

    DstViewType dst;
    SrcViewType src;
};

struct deep_copy_t {
    template <typename DstViewType, typename SrcViewType>
    constexpr auto operator()(DstViewType dst, SrcViewType src) const {
        const size_t shape = dst.size();
        return stdexec::bulk(stdexec::par, shape, DeepCopy{std::move(dst), std::move(src)});
    }

    template <stdexec::sender Sndr, typename DstViewType, typename SrcViewType>
    constexpr auto operator()(Sndr&& sndr, DstViewType dst, SrcViewType src) const {
        return std::forward<Sndr>(sndr) | this->operator()(std::move(dst), std::move(src));
    }
};

inline constexpr deep_copy_t deep_copy{};

template <typename AMatrixType, typename XViewType, typename YViewType>
struct SPMV {
    using scalar_t = typename YViewType::non_const_value_type;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T irow) const {
        scalar_t sum = 0;
        for (auto ientry = mat.graph.row_map(irow); ientry < mat.graph.row_map(irow + 1); ++ientry) {
            sum += mat.values(ientry) * view_x(mat.graph.entries(ientry));
        }

        view_y(irow) = (beta == scalar_t(0)) ? alpha * sum : alpha * sum + beta * view_y(irow);
    }

    scalar_t alpha;
    AMatrixType mat;
    XViewType view_x;
    scalar_t beta;
    YViewType view_y;
};

struct spmv_t {
    template <typename AlphaType, typename AMatrixType, typename XViewType, typename BetaType, typename YViewType>
    constexpr auto
        operator()(AlphaType alpha, AMatrixType mat, XViewType view_x, BetaType beta, YViewType view_y) const {
        const size_t shape = mat.graph.row_map.size() - 1;
        return stdexec::bulk(
            stdexec::par, shape, SPMV{alpha, std::move(mat), std::move(view_x), beta, std::move(view_y)});
    }

    template <
        stdexec::sender Sndr,
        typename AlphaType,
        typename AMatrixType,
        typename XViewType,
        typename BetaType,
        typename YViewType
    >
    constexpr auto
        operator()(Sndr&& sndr, AlphaType alpha, AMatrixType mat, XViewType view_x, BetaType beta, YViewType view_y)
            const {
        return std::forward<Sndr>(sndr)
             | this->operator()(alpha, std::move(mat), std::move(view_x), beta, std::move(view_y));
    }
};

inline constexpr spmv_t spmv{};

template <typename ResultViewType, typename XViewType, typename YViewType>
struct Dot {
    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T irow) const {
        Kokkos::atomic_add(result.data(), view_x(irow) * view_y(irow));
    }

    ResultViewType result;
    XViewType view_x;
    YViewType view_y;
};

struct dot_t {
    template <typename ResultViewType, typename XViewType, typename YViewType>
    constexpr auto operator()(
        ResultViewType result, // NOLINT(performance-unnecessary-value-param)
        XViewType view_x,
        YViewType view_y) const {
        const size_t shape = view_x.size();
        return stdexec::then(KOKKOS_LAMBDA() { result() = 0; })
             | stdexec::bulk(stdexec::par, shape, Dot{result, std::move(view_x), std::move(view_y)});
    }

    template <stdexec::sender Sndr, typename ResultViewType, typename XViewType, typename YViewType>
    constexpr auto operator()(Sndr&& sndr, ResultViewType result, XViewType view_x, YViewType view_y) const {
        return std::forward<Sndr>(sndr) | this->operator()(std::move(result), std::move(view_x), std::move(view_y));
    }
};

inline constexpr dot_t dot{};

template <typename T>
requires(!Kokkos::is_view_v<std::remove_cvref_t<T>>)
KOKKOS_FUNCTION constexpr decltype(auto) get_value(T&& val) {
    return std::forward<T>(val);
}

template <typename T>
requires(Kokkos::is_view_v<std::remove_cvref_t<T>> && std::remove_cvref_t<T>::rank() == 0)
KOKKOS_FUNCTION constexpr decltype(auto) get_value(T&& val) {
    return std::forward<T>(val)();
}

template <typename AMaybeViewType, typename XViewType, typename BMaybeViewType, typename YViewType>
struct Axpby {
    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T irow) const {
        view_y(irow) = get_value(coeff_a) * view_x(irow) + get_value(coeff_b) * view_y(irow);
    }

    AMaybeViewType coeff_a;
    XViewType view_x;
    BMaybeViewType coeff_b;
    YViewType view_y;
};

struct axpby_t {
    template <typename AMaybeViewType, typename XViewType, typename BMaybeViewType, typename YViewType>
    constexpr auto
        operator()(AMaybeViewType coeff_a, XViewType view_x, BMaybeViewType coeff_b, YViewType view_y) const {
        const size_t shape = view_x.size();
        return stdexec::bulk(
            stdexec::par, shape, Axpby{std::move(coeff_a), std::move(view_x), std::move(coeff_b), std::move(view_y)});
    }

    template <
        stdexec::sender Sndr,
        typename AMaybeViewType,
        typename XViewType,
        typename BMaybeViewType,
        typename YViewType
    >
    constexpr auto
        operator()(Sndr&& sndr, AMaybeViewType coeff_a, XViewType view_x, BMaybeViewType coeff_b, YViewType view_y)
            const {
        return std::forward<Sndr>(sndr)
             | this->operator()(std::move(coeff_a), std::move(view_x), std::move(coeff_b), std::move(view_y));
    }
};

inline constexpr axpby_t axpby{};

template <typename MatrixType, typename RhsViewType, typename SolViewType>
stdexec::sender auto
    cg(stdexec::scheduler auto schd,
       stdexec::scheduler auto host_schd,
       MatrixType mat,  // NOLINT(performance-unnecessary-value-param)
       RhsViewType rhs, // NOLINT(performance-unnecessary-value-param)
       SolViewType sol, // NOLINT(performance-unnecessary-value-param)
       typename RhsViewType::non_const_value_type tol) {
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

    const auto dir_mat_dir = Kokkos::subview(scalars, 0);
    const auto alpha = Kokkos::subview(scalars, 1);
    const auto alpha_neg = Kokkos::subview(scalars, 2);
    const auto res_res_new = Kokkos::subview(scalars, 3);
    const auto beta = Kokkos::subview(scalars, 4);

    // clang-format off
    return stdexec::schedule(schd)
        | deep_copy(res, rhs)
        | spmv(scalar_t{-1}, mat, sol, scalar_t{1}, res) // NOLINT(performance-unnecessary-value-param)
        | dot(res_res, res, res)
        | deep_copy(dir, res)
        | stdexec::let_value([=] {
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
                    | stdexec::continues_on(host_schd)
                    | stdexec::then([=] {
                        return Kokkos::sqrt(Kokkos::abs(res_res())) < tol;
                    }));
           });
    // clang-format on
}

} // namespace Examples::CG

#endif // KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
