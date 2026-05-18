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

//! Inspired by https://github.com/NVIDIA/stdexec/blob/8c5eedd0fcf9a8ebcdb75d988f72f88efcf64a37/include/stdexec/__detail/__completion_behavior.hpp#L33.
struct GraphComposition {
    //! Attach to the existing graph of the predecessor.
    struct Attach { };

    //! Create a new graph and attach after the root node.
    struct Create { };

    //! Use the @ref Attach policy if @p QueryableA or @p QueryableB is queryable with @ref get_node_t.
    template <typename QueryableA, typename QueryableB>
    using policy_t = std::conditional_t<
        stdexec::__queryable_with<QueryableA, get_node_t> || stdexec::__queryable_with<QueryableB, get_node_t>,
        Attach,
        Create
    >;

    template <typename Exec, typename QueryableA, typename QueryableB>
    struct node_helper_t;

    template <typename Exec, typename QueryableA, typename QueryableB>
    requires(std::same_as<policy_t<QueryableA, QueryableB>, Create>)
    struct node_helper_t<Exec, QueryableA, QueryableB> {
        using type = typename Kokkos::Experimental::Graph<Exec>::root_t;
    };

    template <typename Exec, typename QueryableA, typename QueryableB>
    requires(
        std::same_as<policy_t<QueryableA, QueryableB>, Attach> && stdexec::__queryable_with<QueryableA, get_node_t>)
    struct node_helper_t<Exec, QueryableA, QueryableB> {
        using type = stdexec::__query_result_t<QueryableA, get_node_t>;
    };

    template <typename Exec, typename QueryableA, typename QueryableB>
    requires(
        std::same_as<policy_t<QueryableA, QueryableB>, Attach> && stdexec::__queryable_with<QueryableB, get_node_t>)
    struct node_helper_t<Exec, QueryableA, QueryableB> {
        using type = stdexec::__query_result_t<QueryableB, get_node_t>;
    };

    /**
     * If @ref policy_t is @ref Create, it is set to the type of the root node.
     * Otherwise, it is set as the query result of @p QueryableA or @p QueryableB for @ref get_node_t.
     */
    template <typename Exec, typename QueryableA, typename QueryableB>
    using node_t = typename node_helper_t<Exec, QueryableA, QueryableB>::type;
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_GET_GRAPH_HPP
