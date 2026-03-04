#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONCEPTS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONCEPTS_HPP

#include <concepts>

namespace Kokkos::Experimental::graph::details
{
/// @todo This constraint ain't clear but should support the case of chaining
///       without an underlying @c Kokkos::Graph.
template <typename Sender>
concept is_graph_sender = ! Kokkos::is_execution_space_v<Sender>;

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONCEPTS_HPP
