#ifndef KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_HPP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.graph_impl;
#else
#    include "Kokkos_Graph.hpp"
#endif

namespace Tests::Utils {

//! Type of an aggregate node.
template <Kokkos::ExecutionSpace Exec>
using aggregate_node_t =
    Kokkos::Experimental::GraphNodeRef<Exec, typename Kokkos::Impl::GraphImpl<Exec>::aggregate_impl_t>;

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_HPP
