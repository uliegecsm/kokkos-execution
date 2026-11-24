#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP

#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Formatters/TxtFormatter.h"
#    include "plog/Initializers/ConsoleInitializer.h"
#    include "plog/Log.h"
#endif

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "tests/utils/functors/throws_when_copied.hpp"
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
    using view_um_s_t = Kokkos::View<value_t, Kokkos::SharedSpace, Kokkos::MemoryUnmanaged>;

    static constexpr bool on_device = Tests::Utils::on_device<Exec>();

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
#    define KOKKOS_EXECUTION_THREADS_THROWS_WHAT "Called by a worker thread, can only be called by the master process."

#    define KOKKOS_EXECUTION_THREADS_THROWS_SKIP_MSG                                                                   \
        "Kokkos::Threads parallel regions must be launched from the thread that initialized Kokkos."

#    define KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT(_sndr_)                                                \
        ASSERT_THAT(                                                                                                   \
            Tests::Utils::Functors::MutableMoveToSyncWait{.sndr = std::move(_sndr_)},                                  \
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(KOKKOS_EXECUTION_THREADS_THROWS_WHAT)));

#    define KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(_sndr_)                                       \
        if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::Threads>) {                                           \
            KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT(_sndr_)                                                \
            GTEST_SKIP() << KOKKOS_EXECUTION_THREADS_THROWS_SKIP_MSG;                                                  \
        }

#    define KOKKOS_EXECUTION_RECORD_EVENTS_BUT_THREADS_MAY_THROW_ON_SYNC_WAIT_SKIP(_recorder_listener_type_, _sndr_)   \
        std::deque<typename _recorder_listener_type_::event_variant_t> recorded_events;                                \
        try {                                                                                                          \
            recorded_events = Tests::Utils::record_sync_wait<_recorder_listener_type_>(std::move(_sndr_));             \
        } catch (const std::runtime_error& err) {                                                                      \
            if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::Threads>) {                                       \
                ASSERT_THAT(err.what(), testing::HasSubstr(KOKKOS_EXECUTION_THREADS_THROWS_WHAT));                     \
                GTEST_SKIP() << KOKKOS_EXECUTION_THREADS_THROWS_SKIP_MSG;                                              \
            } else {                                                                                                   \
                throw;                                                                                                 \
            }                                                                                                          \
        }
#else
#    define KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(_sndr_)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define KOKKOS_EXECUTION_RECORD_EVENTS_BUT_THREADS_MAY_THROW_ON_SYNC_WAIT_SKIP(_recorder_listener_type_, _sndr_)   \
        const auto recorded_events = Tests::Utils::record_sync_wait<_recorder_listener_type_>(std::move(_sndr_));
#endif

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP
