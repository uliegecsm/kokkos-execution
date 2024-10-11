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

template <typename ViewType>
struct MyDummyFunctor
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const { ++data(index); }
};

//! Dummy function that can be transparently used with graph or execution space instance.
template <typename Sender, typename ViewType>
decltype(auto) my_function(Sender&& sender, const ViewType& data)
{
    using policy_t = Kokkos::RangePolicy<typename std::remove_reference_t<Sender>::execution_space>;

    return std::forward<Sender>(sender) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, 1),
        MyDummyFunctor{.data = data}
    );
}

class ParallelForTest : public ::testing::Test
{
public:
    using view_t = Kokkos::View<int[1], execution_space, Kokkos::MemoryTraits<Kokkos::Atomic>>;

public:
    void SetUp() override
    {
        this->exec = execution_space {};
        this->data = view_t(Kokkos::view_alloc("data", exec));
    }
protected:
    execution_space exec;
    view_t          data;
};

#define CHECK_DATA_CONTENT ASSERT_EQ(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data)(0), 1);

//! @test Check the execution space instance mode for the parallel-for construct.
TEST_F(ParallelForTest, exec)
{
    decltype(auto) tail = my_function(exec, data);

    static_assert(std::same_as<decltype(tail), execution_space&>);

    ASSERT_EQ(std::addressof(exec), std::addressof(tail)) << "You abused of the execution space instance.";

    Kokkos::Experimental::submit(exec, std::move(tail));

    exec.fence();

    CHECK_DATA_CONTENT
}

//! @test Check the graph mode for the parallel-for construct.
TEST_F(ParallelForTest, graph)
{
    decltype(auto) root = Kokkos::Experimental::graph::create_graph(exec);

    decltype(auto) tail = my_function(root, data);

    static_assert(Kokkos::Impl::is_specialization_of<decltype(tail), Kokkos::Experimental::GraphNodeRef>::value);

    Kokkos::Experimental::graph::submit(exec, std::move(tail));

    exec.fence();

    CHECK_DATA_CONTENT
}

} // namespace tests::kokkos_ext
