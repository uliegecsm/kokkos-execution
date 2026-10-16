#ifndef KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
#define KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Examples::CG {

template <typename OrdinalType, Kokkos::ExecutionSpace Exec>
struct CrsGraph {
    using ordinal_t = OrdinalType;

    using row_map_t = Kokkos::View<const size_t*, Exec>;
    using entries_t = Kokkos::View<ordinal_t*, Exec>;

    row_map_t row_map;
    entries_t entries;
};

template <typename ScalarType, typename OrdinalType, Kokkos::ExecutionSpace Exec>
struct CrsMatrix {
    using scalar_t = ScalarType;
    using ordinal_t = OrdinalType;

    using graph_t = CrsGraph<ordinal_t, Exec>;
    using values_t = Kokkos::View<scalar_t*, Exec>;

    using const_t = CrsMatrix<const scalar_t, ordinal_t, Exec>;

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

template <typename XViewType, typename YViewType>
struct Dot {
    KOKKOS_FUNCTION void operator()(const typename XViewType::execution_space::index_type irow, typename XViewType::value_type& value) const {
        value += view_x(irow) * view_y(irow);
    }

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
        return Kokkos::Execution::parallel_reduce(Kokkos::RangePolicy<typename XViewType::execution_space>(0, shape), Dot{std::move(view_x), std::move(view_y)}, std::move(result));
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

} // namespace Examples::CG

#endif // KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
