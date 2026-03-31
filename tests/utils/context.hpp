#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP

#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Formatters/TxtFormatter.h"
#    include "plog/Initializers/ConsoleInitializer.h"
#    include "plog/Log.h"
#endif

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "tests/utils/kokkos.hpp"

namespace Tests::Utils {

template <template <typename> typename ContextType, Kokkos::ExecutionSpace Exec>
struct ContextTest
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<Exec> {
   public:
    using context_t = ContextType<Exec>;
    using scheduler_t = decltype(std::declval<const context_t>().get_scheduler());
    using schedule_sender_t = decltype(stdexec::schedule(std::declval<scheduler_t>()));

    using value_t = int;
    using view_s_t = Kokkos::View<value_t, Kokkos::SharedSpace>;

    static constexpr bool on_device = Tests::Utils::on_device<TEST_EXECUTION_SPACE>();

   public:
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
    static void SetUpTestSuite() {
        plog::init<plog::TxtFormatter>(plog::debug, plog::streamStdOut);
    }
#endif
};

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

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP
