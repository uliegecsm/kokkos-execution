#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

/**
 * @addtogroup unittests
 *
 * @c Kokkos extension utilities
 * -----------------------------
 *
 * This group of tests check the utilities found in @ref kokkos_ext/impl/Utils.hpp.
 *
 * The tests can be found in @ref kokkos_ext/test_utils.cpp.
 */

namespace tests::kokkos_ext
{

using execution_space = Kokkos::DefaultExecutionSpace;

/**
 * @test Check that @ref Kokkos::Experimental::graph::details::update_policy
 *       works when updating the execution space instance of a @c Kokkos::RangePolicy.
 */
TEST(update_policy, range_policy_exec)
{
    const auto execs = Kokkos::Experimental::partition_space(execution_space{}, 1, 1);

    ASSERT_NE(execs.at(0), execs.at(1));

    const Kokkos::RangePolicy<execution_space> initial(execs.at(0), 0, 1, Kokkos::ChunkSize(2));
    ASSERT_EQ(initial.space(), execs.at(0));

    const auto updated = Kokkos::Experimental::graph::details::update_policy(initial, execs.at(1));
    ASSERT_EQ(updated.space(),      execs.at(1));
    ASSERT_EQ(updated.begin(),      0);
    ASSERT_EQ(updated.end(),        1);
    ASSERT_EQ(updated.chunk_size(), 2);
}

} // namespace tests::kokkos_ext
