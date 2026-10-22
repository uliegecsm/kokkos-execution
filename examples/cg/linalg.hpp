#ifndef KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
#define KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Examples::CG {

/**
 * @brief Compressed row storage (CRS) graph data structures.
 *
 * Minimalistic implementation inspired by the design of the fully featured implementation of @c KokkosSparse::StaticCrsGraph.
 *
 * See also:
 * - https://github.com/kokkos/kokkos-kernels/blob/505a9c982e8c06f4619d6a80c8f474f9ce7b8e49/sparse/src/KokkosSparse_StaticCrsGraph.hpp#L222-L245
 * - https://github.com/kokkos/kokkos-kernels/blob/505a9c982e8c06f4619d6a80c8f474f9ce7b8e49/common/src/KokkosKernels_default_types.hpp#L36-L45
 * - https://github.com/kokkos/kokkos-kernels/blob/505a9c982e8c06f4619d6a80c8f474f9ce7b8e49/cmake/kokkoskernels_eti_offsets.cmake#L4
 */
template <typename OrdinalType, Kokkos::MemorySpace Mem>
struct CrsGraph {
    using ordinal_t = OrdinalType;

    using row_map_t = Kokkos::View<const size_t*, Mem>;
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

template <typename DstViewType, typename SrcViewType>
DeepCopy(DstViewType, SrcViewType) -> DeepCopy<DstViewType, SrcViewType>;

struct deep_copy_t {
    template <typename DstViewType, typename SrcViewType>
    constexpr auto operator()(DstViewType&& dst, SrcViewType&& src) const {
        return stdexec::__closure(*this, std::forward<DstViewType>(dst), std::forward<SrcViewType>(src));
    }

    template <stdexec::sender Sndr, typename DstViewType, typename SrcViewType>
    constexpr auto operator()(Sndr&& sndr, DstViewType&& dst, SrcViewType&& src) const {
        const size_t shape = dst.size();
        return std::forward<Sndr>(sndr)
             | Kokkos::Execution::parallel_for(
                   "parallel_for (deep_copy)",
                   shape,
                   DeepCopy{std::forward<DstViewType>(dst), std::forward<SrcViewType>(src)});
    }
};

//template <typename Derived>
//struct PipeableSenderAdaptorBase {
//    template <typename... Args>
//    constexpr auto operator()(Args&&... args) const {
//        return stdexec::__closure(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
//    }
//};
//
//struct deep_copy_t : public PipeableSenderAdaptorBase<deep_copy_t> {
//    using PipeableSenderAdaptorBase<deep_copy_t>::operator();
//
//    template <stdexec::sender Sndr, typename DstViewType, typename SrcViewType>
//    constexpr auto operator()(Sndr&& sndr, DstViewType&& dst, SrcViewType&& src) const {
//        const size_t shape = dst.size();
//        return std::forward<Sndr>(sndr)
//             | Kokkos::Execution::parallel_for(
//                   "parallel_for (deep_copy)",
//                   shape,
//                   DeepCopy{std::forward<DstViewType>(dst), std::forward<SrcViewType>(src)});
//    }
//};

//template <typename Derived>
//struct Base {
//    template <stdexec::sender Sndr, typename... Args> requires std::invocable<const Derived&, Args...>
//    constexpr auto operator()(Sndr&& sndr, Args&&... args) const {
//        return std::forward<Sndr>(sndr) | static_cast<const Derived&>(*this)(std::forward<Args>(args)...);
//    }
//};
//
//struct deep_copy_t : public Base<deep_copy_t> {
//    template <typename DstViewType, typename SrcViewType> requires (!stdexec::sender<DstViewType>)
//    constexpr auto operator()(DstViewType&& dst, SrcViewType&& src) const {
//        const size_t shape = dst.size();
//        return Kokkos::Execution::parallel_for(
//            "parallel_for (deep_copy)", shape, DeepCopy{std::forward<DstViewType>(dst), std::forward<SrcViewType>(src)});
//    }
//};

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

template <typename AMatrixType, typename XViewType, typename YViewType>
SPMV(
    typename YViewType::non_const_value_type,
    AMatrixType,
    XViewType,
    typename YViewType::non_const_value_type,
    YViewType) -> SPMV<AMatrixType, XViewType, YViewType>;

struct spmv_t {
    template <typename AlphaType, typename AMatrixType, typename XViewType, typename BetaType, typename YViewType>
    constexpr auto
        operator()(AlphaType alpha, AMatrixType&& mat, XViewType&& view_x, BetaType beta, YViewType&& view_y) const {
        return stdexec::__closure(
            *this,
            alpha,
            std::forward<AMatrixType>(mat),
            std::forward<XViewType>(view_x),
            beta,
            std::forward<YViewType>(view_y));
    }

    template <
        stdexec::sender Sndr,
        typename AlphaType,
        typename AMatrixType,
        typename XViewType,
        typename BetaType,
        typename YViewType
    >
    constexpr auto operator()(
        Sndr&& sndr,
        AlphaType alpha,
        AMatrixType&& mat,
        XViewType&& view_x,
        BetaType beta,
        YViewType&& view_y) const {
        const size_t shape = mat.graph.row_map.size() - 1;
        return std::forward<Sndr>(sndr)
             | Kokkos::Execution::parallel_for(
                   "parallel_for (spmv)",
                   shape,
                   SPMV{
                       alpha,
                       std::forward<AMatrixType>(mat),
                       std::forward<XViewType>(view_x),
                       beta,
                       std::forward<YViewType>(view_y)});
    }
};

inline constexpr spmv_t spmv{};

template <typename XViewType, typename YViewType>
struct Dot {
    KOKKOS_FUNCTION void operator()(const auto irow, typename XViewType::value_type& value) const {
        value += view_x(irow) * view_y(irow);
    }

    XViewType view_x;
    YViewType view_y;
};

template <typename XViewType, typename YViewType>
Dot(XViewType, YViewType) -> Dot<XViewType, YViewType>;

struct dot_t {
    template <typename ResultViewType, typename XViewType, typename YViewType>
    constexpr auto operator()(ResultViewType&& result, XViewType&& view_x, YViewType&& view_y) const {
        return stdexec::__closure(
            *this,
            std::forward<ResultViewType>(result),
            std::forward<XViewType>(view_x),
            std::forward<YViewType>(view_y));
    }

    template <typename Sndr, typename ResultViewType, typename XViewType, typename YViewType>
    constexpr auto operator()(
        Sndr&& sndr,
        ResultViewType&& result, // NOLINT(performance-unnecessary-value-param)
        XViewType&& view_x,
        YViewType&& view_y) const {
        const size_t shape = view_x.size();
        return std::forward<Sndr>(sndr)
             | Kokkos::Execution::parallel_reduce(
                   "parallel_reduce (dot)",
                   shape,
                   Dot{std::forward<XViewType>(view_x), std::forward<YViewType>(view_y)},
                   std::forward<ResultViewType>(result));
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

template <typename AMaybeViewType, typename XViewType, typename BMaybeViewType, typename YViewType>
Axpby(AMaybeViewType, XViewType, BMaybeViewType, YViewType)
    -> Axpby<AMaybeViewType, XViewType, BMaybeViewType, YViewType>;

struct axpby_t {
    template <typename AMaybeViewType, typename XViewType, typename BMaybeViewType, typename YViewType>
    constexpr auto
        operator()(AMaybeViewType&& coeff_a, XViewType&& view_x, BMaybeViewType&& coeff_b, YViewType&& view_y) const {
        return stdexec::__closure(
            *this,
            std::forward<AMaybeViewType>(coeff_a),
            std::forward<XViewType>(view_x),
            std::forward<BMaybeViewType>(coeff_b),
            std::forward<YViewType>(view_y));
    }

    template <
        stdexec::sender Sndr,
        typename AMaybeViewType,
        typename XViewType,
        typename BMaybeViewType,
        typename YViewType
    >
    constexpr auto operator()(
        Sndr&& sndr,
        AMaybeViewType&& coeff_a,
        XViewType&& view_x,
        BMaybeViewType&& coeff_b,
        YViewType&& view_y) const {
        const size_t shape = view_x.size();
        return std::forward<Sndr>(sndr)
             | Kokkos::Execution::parallel_for(
                   "parallel_for (axbpy)",
                   shape,
                   Axpby{
                       std::forward<AMaybeViewType>(coeff_a),
                       std::forward<XViewType>(view_x),
                       std::forward<BMaybeViewType>(coeff_b),
                       std::forward<YViewType>(view_y)});
    }
};

inline constexpr axpby_t axpby{};

} // namespace Examples::CG

#endif // KOKKOS_EXECUTION_EXAMPLES_CG_LINALG_HPP
