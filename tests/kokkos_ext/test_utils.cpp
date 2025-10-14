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
    const auto [exec_a, exec_b] = Kokkos::Experimental::partition_space(execution_space{}, 1, 1);

    /// For @c Kokkos::OpenMP, see https://github.com/kokkos/kokkos/commit/a09c6ce45655f37bedf767d68ff42b7382ba89e7.
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::same_as<execution_space, Kokkos::OpenMP>) {
        ASSERT_EQ(exec_a, exec_b);
    } else {
#endif
        ASSERT_NE(exec_a, exec_b);
#if defined(KOKKOS_ENABLE_OPENMP)
    }
#endif

    const Kokkos::RangePolicy<execution_space> initial(exec_a, 0, 1, Kokkos::ChunkSize(2));
    ASSERT_EQ(initial.space(), exec_a);

    const auto updated = Kokkos::Experimental::graph::details::update_policy(initial, exec_b);
    ASSERT_EQ(updated.space(),      exec_b);
    ASSERT_EQ(updated.begin(),      0);
    ASSERT_EQ(updated.end(),        1);
    ASSERT_EQ(updated.chunk_size(), 2);
}

} // namespace tests::kokkos_ext
