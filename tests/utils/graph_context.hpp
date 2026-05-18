#ifndef KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP

#include "Kokkos_Graph_fwd.hpp"
#include "impl/Kokkos_DeviceHandle.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "tests/utils/context.hpp"

namespace Tests::Utils {

template <Kokkos::ExecutionSpace Exec>
struct GraphContextTest : public ContextTest<Kokkos::Execution::GraphContext, Exec> {
    using device_handle_t = Kokkos::Impl::DeviceHandle<Exec>;
    using graph_t = Kokkos::Experimental::Graph<Exec>;

    device_handle_t device_handle{this->exec};
};

/**
 * Use this macro when some data used in the graph may still be initialized by @c Kokkos
 * on a given execution space instance, while the graph is submitted on another execution
 * space instance.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(_exec_) _exec_.fence("waiting for the data to be ready")

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP
