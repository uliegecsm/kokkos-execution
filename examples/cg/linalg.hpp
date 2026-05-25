#ifndef KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
#define KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP

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
    constexpr auto operator()(ResultViewType result, XViewType view_x, YViewType view_y)
        const { // NOLINT(performance-unnecessary-value-param)
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

template <typename MatrixType, typename RhsViewType, typename SolViewType>
stdexec::sender auto
    cg(stdexec::scheduler auto schd,
       MatrixType mat,
       RhsViewType rhs,   // NOLINT(performance-unnecessary-value-param)
       SolViewType sol) { // NOLINT(performance-unnecessary-value-param)
    using scalar_t = typename RhsViewType::non_const_value_type;

    const typename RhsViewType::non_const_type res(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "res"), rhs.size());
    const Kokkos::View<scalar_t, Kokkos::SharedHostPinnedSpace> res_dot_old(
        Kokkos::view_alloc(schd.state->exec, Kokkos::WithoutInitializing, "res dot old"));

    return stdexec::schedule(schd) 
         | deep_copy(res, rhs)
         | spmv(scalar_t{-1}, mat, sol, scalar_t{1}, res) // NOLINT(performance-unnecessary-value-param)
         | dot(res_dot_old, res, res);
}

} // namespace Examples::CG

#endif // KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
