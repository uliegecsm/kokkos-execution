#ifndef KOKKOS_EXECUTION_GRAPH_GET_NODE_HPP
#define KOKKOS_EXECUTION_GRAPH_GET_NODE_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::GraphImpl {

/**
 * Query an object for its graph node.
 *
 * See also https://github.com/NVIDIA/cccl/blob/6e592beda9c50aeb3cc62dd1036d509f540ccbe7/libcudacxx/include/cuda/__stream/get_stream.h.
 */
struct get_node_t : public stdexec::__query<get_node_t> { };

inline constexpr get_node_t get_node{};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_GET_NODE_HPP
