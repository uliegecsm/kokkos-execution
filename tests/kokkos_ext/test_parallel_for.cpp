#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/kokkos_ext/Helpers.hpp"

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

//! Dummy function that can be transparently used with graph or execution space instance.
template <bool label, typename Sender, typename ViewType>
decltype(auto) my_function(Sender&& sender, const ViewType& data)
{
    using policy_t = Kokkos::RangePolicy<typename std::remove_reference_t<Sender>::execution_space>;

    #define MY_FUNCTION_CORE(...)                                                        \
        return std::forward<Sender>(sender) | Kokkos::Experimental::graph::parallel_for( \
            __VA_ARGS__ __VA_OPT__(,)                                                    \
            policy_t(0, 1),                                                              \
            MyDummyFunctor{.data = data}                                                 \
        );
    if constexpr (label) MY_FUNCTION_CORE("this is a test parallel-for")
    else                 MY_FUNCTION_CORE()
}

template <typename T>
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

using ParallelForTestTypes = ::testing::Types<
    std::integral_constant<bool, true>,
    std::integral_constant<bool, false>
>;

TYPED_TEST_SUITE(ParallelForTest, ParallelForTestTypes);

//! Avoid any global fence, as it could hide potential issues.
#define CHECK_DATA_CONTENT(__exec__)                                     \
    typename TestFixture::view_t::value_type result = 0;                 \
    Kokkos::deep_copy(__exec__, result, Kokkos::subview(this->data, 0)); \
    __exec__.fence();                                                    \
    ASSERT_EQ(result, 1);

//! @test Check the execution space instance mode for the parallel-for construct.
TYPED_TEST(ParallelForTest, exec)
{
    decltype(auto) tail = my_function<TypeParam::value>(this->exec, this->data);

    static_assert(std::same_as<decltype(tail), execution_space&>);

    ASSERT_EQ(std::addressof(this->exec), std::addressof(tail)) << "You abused of the execution space instance.";

    Kokkos::Experimental::submit(this->exec, std::move(tail));

    CHECK_DATA_CONTENT(this->exec)
}

//! @test Check the graph mode for the parallel-for construct.
TYPED_TEST(ParallelForTest, graph)
{
    decltype(auto) root = Kokkos::Experimental::graph::create_graph(this->exec);

    decltype(auto) tail = my_function<TypeParam::value>(root, this->data);

    static_assert(Kokkos::Impl::is_specialization_of<decltype(tail), Kokkos::Experimental::GraphNodeRef>::value);

    Kokkos::Experimental::graph::submit(this->exec, std::move(tail));

    CHECK_DATA_CONTENT(this->exec)
}

} // namespace tests::kokkos_ext
