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

    //! Use the @ref Attach policy if @p Queryable is queryable with @ref get_node_t.
    template <typename Queryable>
    using policy_t = std::conditional_t<stdexec::__queryable_with<Queryable, get_node_t>, Attach, Create>;

    template <typename Exec, typename Queryable>
    struct node_helper_t;

    template <typename Exec, typename Queryable>
    requires(std::same_as<policy_t<Queryable>, Create>)
    struct node_helper_t<Exec, Queryable> {
        using type = typename Kokkos::Experimental::Graph<Exec>::root_t;
    };

    template <typename Exec, typename Queryable>
    requires(std::same_as<policy_t<Queryable>, Attach>)
    struct node_helper_t<Exec, Queryable> {
        using type = stdexec::__query_result_t<Queryable, get_node_t>;
    };

    /**
     * If @ref policy_t for @p Queryable is the @ref Create policy, it is set to the type of the root node.
     * Otherwise, it is set as the query result of @p Queryable for @ref get_node_t.
     */
    template <typename Exec, typename Queryable>
    using node_t = typename node_helper_t<Exec, Queryable>::type;
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_GET_GRAPH_HPP
