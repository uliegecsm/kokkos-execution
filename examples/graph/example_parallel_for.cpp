#include "kokkos-execution/graph.hpp"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

/**
 * @addtogroup examples
 *
 * @example examples/graph/example_parallel_for.cpp
 *
 * A sequence of @c parallel_for nodes in a @c Kokkos::Graph
 * ---------------------------------------------------------
 *
 * This example creates a sequence of @c parallel_for nodes using either the @c Kokkos API,
 * or the @ref Kokkos::Execution::GraphContext.
 */

namespace Examples::Graph {

template <char Value, typename ViewType>
struct Work {
    ViewType data;

    template <std::integral IndexType>
    KOKKOS_FUNCTION void operator()(const IndexType index) const noexcept {
        data(index) = Value;
    }
};

template <Kokkos::ExecutionSpace Exec>
struct ParallelFor : public Kokkos::utils::tests::scoped::ExecutionSpace<Exec> {
   public:
    using base_t = Kokkos::utils::tests::scoped::ExecutionSpace<Exec>;
    using policy_t = Kokkos::RangePolicy<Exec>;
    using view_t = Kokkos::View<char[6], Exec>;

   public:
    ParallelFor()
        : base_t()
        , data(Kokkos::view_alloc(Kokkos::WithoutInitializing, this->exec, "data")) {
    }

    void check() const {
        const auto mirror = Kokkos::create_mirror_view_and_copy(data);
        if (std::strcmp(mirror.data(), "ABCBA") != 0) {
            throw std::runtime_error(std::format("'{}' is not equal to 'ABCBA'.", mirror.data()));
        }
    }

    void reset() const {
        Kokkos::deep_copy(this->exec, data, 0);
    }

    //! Create the sequence with @c Kokkos::Graph.
    void with_kokkos_graph() const {
        reset();

        const auto graph = Kokkos::Experimental::create_graph(
            Kokkos::Experimental::get_device_handle(this->exec), [&](const auto& root) {
                root.then_parallel_for(policy_t(0, 5), Work<'A', view_t>{.data = data})
                    .then_parallel_for(policy_t(1, 4), Work<'B', view_t>{.data = data})
                    .then_parallel_for(policy_t(2, 3), Work<'C', view_t>{.data = data});
            });

        graph.submit(this->exec);
        this->exec.fence();

        check();
    }

    //! Create the sequence with @ref Kokkos::Execution::GraphContext.
    void with_graph_context() const {
        reset();

        const Kokkos::Execution::GraphContext<Exec> gctx{this->exec};

        stdexec::sync_wait(
            stdexec::schedule(gctx.get_scheduler())
            | Kokkos::Execution::parallel_for(policy_t(0, 5), Work<'A', view_t>{.data = data})
            | Kokkos::Execution::parallel_for(policy_t(1, 4), Work<'B', view_t>{.data = data})
            | Kokkos::Execution::parallel_for(policy_t(2, 3), Work<'C', view_t>{.data = data}));

        check();
    }

   protected:
    view_t data;
};


} // namespace Examples::Graph

int main(int argc, char* argv[]) {
    const Kokkos::ScopeGuard guard{argc, argv};
    {
        const Examples::Graph::ParallelFor<EXAMPLE_EXECUTION_SPACE> pfor{};

        pfor.with_kokkos_graph();
        pfor.with_graph_context();
    }
    return EXIT_SUCCESS;
}
