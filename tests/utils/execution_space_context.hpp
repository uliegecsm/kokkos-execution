#ifndef KOKKOS_EXECUTION_TESTS_UTILS_EXECUTION_SPACE_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_EXECUTION_SPACE_CONTEXT_HPP

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/context.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"

namespace Tests::Utils {

template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpaceContextTest : public ContextTest<Kokkos::Execution::ExecutionSpaceContext, Exec> { };

#if defined(KOKKOS_ENABLE_THREADS)
/**
 * According to
 * https://github.com/kokkos/kokkos/blob/301c37189a7fef46e68768ad9df160113f7ea052/core/unit_test/TestExecSpaceThreadSafety.hpp#L79,
 * any @c Kokkos parallel region launch for @c Kokkos::Threads has to be done from the thread that initialized @c Kokkos.
 * Otherwise, it triggers
 * https://github.com/kokkos/kokkos/blob/301c37189a7fef46e68768ad9df160113f7ea052/core/src/Threads/Kokkos_Threads_Instance.cpp#L91.
 *
 * However, the thread that initialized @c Kokkos is unknown to @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * Therefore, @ref Kokkos::Execution::ExecutionSpaceContext gives the same user experience as raw @c Kokkos: calling
 * a parallel dispatch from another thread will throw.
 */
#    define KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT(_sndr_)                                                \
        ASSERT_THAT(                                                                                                   \
            Tests::Utils::Functors::MutableMoveToSyncWait{.sndr = std::move(_sndr_)},                                  \
            ::testing::ThrowsMessage<std::runtime_error>(                                                              \
                ::testing::HasSubstr("Called by a worker thread, can only be called by the master process.")));
#    define KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(_sndr_)                                       \
        if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::Threads>) {                                           \
            KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT(_sndr_)                                                \
            GTEST_SKIP()                                                                                               \
                << "Kokkos::Threads parallel regions must be launched from the thread that initialized " "Kokkos.";    \
        }
#else
#    define KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(_sndr_)
#endif

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_EXECUTION_SPACE_CONTEXT_HPP
