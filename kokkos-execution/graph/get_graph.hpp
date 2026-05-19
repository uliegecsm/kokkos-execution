#ifndef KOKKOS_EXECUTION_GRAPH_GET_GRAPH_HPP
#define KOKKOS_EXECUTION_GRAPH_GET_GRAPH_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/get_node.hpp"

namespace Kokkos::Execution::GraphImpl {

/**
 * Query an object for its graph.
 *
 * See also https://github.com/NVIDIA/cccl/blob/6e592beda9c50aeb3cc62dd1036d509f540ccbe7/libcudacxx/include/cuda/__stream/get_stream.h.
 */
struct get_graph_t : public stdexec::__query<get_graph_t> { };

inline constexpr get_graph_t get_graph{};

#if defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU == 1420 || KOKKOS_COMPILER_GNU == 1520)
//! @bug May be related to https://gcc.gnu.org/legacy-ml/gcc-bugs/2019-01/msg01795.html.
template <typename... Booleans>
struct gcc_mor_helper {
    static constexpr bool value = (Booleans::value || ...);
};

template <typename... Booleans>
static constexpr bool gcc_mor_v = gcc_mor_helper<Booleans...>::value;
#endif

//! Inspired by https://github.com/NVIDIA/stdexec/blob/8c5eedd0fcf9a8ebcdb75d988f72f88efcf64a37/include/stdexec/__detail/__completion_behavior.hpp#L33.
struct GraphComposition {
    //! Attach to the existing graph of the predecessor.
    struct Attach { };

    //! Create a new graph and attach after the root node.
    struct Create { };

    //! Use the @ref Attach policy if any @p Queryables is queryable with @ref get_node_t.
    template <typename... Queryables>
    using policy_t = std::conditional_t<
#if defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU == 1420 || KOKKOS_COMPILER_GNU == 1520)
        gcc_mor_v<stdexec::__mbool<stdexec::__queryable_with<Queryables, get_node_t>>...>,
#else
        (stdexec::__queryable_with<Queryables, get_node_t> || ...),
#endif
        Attach,
        Create
    >;

    template <typename Exec, typename... Queryables>
    struct node_helper_t;

    template <typename Exec, typename... Queryables>
    requires std::same_as<policy_t<Queryables...>, Create>
    struct node_helper_t<Exec, Queryables...> {
        using type = typename Kokkos::Experimental::Graph<Exec>::root_t;
    };

    template <typename Exec, typename FirstQueryable, typename... RestOfQueryables>
    requires(
        std::same_as<policy_t<FirstQueryable, RestOfQueryables...>, Attach>
        && stdexec::__queryable_with<FirstQueryable, get_node_t>)
    struct node_helper_t<Exec, FirstQueryable, RestOfQueryables...> {
        using type = stdexec::__query_result_t<FirstQueryable, get_node_t>;
    };

    template <typename Exec, typename FirstQueryable, typename... RestOfQueryables>
    requires(std::same_as<policy_t<FirstQueryable, RestOfQueryables...>, Attach>)
    struct node_helper_t<Exec, FirstQueryable, RestOfQueryables...> {
        using type = typename node_helper_t<Exec, RestOfQueryables...>::type;
    };

    /**
     * If @ref policy_t is @ref Create, it is set to the type of the root node.
     * Otherwise, it is set as the first valid @ref get_node_t query result in the @p Queryables.
     */
    template <typename Exec, typename... Queryables>
    using node_t = typename node_helper_t<Exec, Queryables...>::type;
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_GET_GRAPH_HPP
