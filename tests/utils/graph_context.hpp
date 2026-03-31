#ifndef KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP

#include "tests/utils/context.hpp"

#include "kokkos-execution/graph.hpp"

namespace Tests::Utils {

template <Kokkos::ExecutionSpace Exec>
struct GraphContextTest : public ContextTest<Kokkos::Execution::GraphContext, Exec> { };

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP
