#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

/**
 * @addtogroup unittests
 *
 * @c Kokkos extensions for graph-compatible parallel-for construct
 * ----------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos_Graph_Execution.hpp effectively
 * makes it possible to use a parallel-for construct in a templated code in either
 * graph or execution space instance mode transparently.
 *
 * The tests can be found in @ref kokkos_ext/test_parallel_for.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

struct MyDummyFunctor
{
    KOKKOS_FUNCTION
    void operator()(const int) const {}
};

//! Dummy function that can be transparently used with graph or execution space instance.
template <typename Sender>
decltype(auto) my_function(Sender&& sender)
{
    using policy_t = Kokkos::RangePolicy<typename std::remove_reference_t<Sender>::execution_space>;

    return std::forward<Sender>(sender) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        MyDummyFunctor{}
    );
}

//! @test Check the execution space instance mode for the parallel-for construct.
TEST(parallel_for, exec)
{
    const execution_space exec {};

    decltype(auto) tail = my_function(exec);

    static_assert(std::same_as<decltype(tail), const execution_space&>);

    ASSERT_EQ(std::addressof(exec), std::addressof(tail)) << "You abused of the execution space instance.";

    Kokkos::Experimental::submit(exec, std::move(tail));

    exec.fence();
}

//! @test Check the graph mode for the parallel-for construct.
TEST(parallel_for, graph)
{
    const execution_space exec {};

    decltype(auto) root = Kokkos::Experimental::graph::create_graph(exec);

    decltype(auto) tail = my_function(root);

    static_assert(Kokkos::Impl::is_specialization_of<decltype(tail), Kokkos::Experimental::GraphNodeRef>::value);

    Kokkos::Experimental::graph::submit(exec, std::move(tail));

    exec.fence();
}

} // namespace tests::kokkos_ext
