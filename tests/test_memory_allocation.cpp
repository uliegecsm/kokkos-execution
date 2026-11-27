#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
#include "kokkos-execution/stdexec/P3955/lifetime.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"
#include "kokkos-execution/memory_allocation.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/sink_receiver.hpp"

/**
 * some blas
 */

// can we find a kokkos kernels function that creates buffers, and when the function scope ends, buffers are destroyed AFTER a global fence ?

namespace Tests {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(MemoryAllocationTest)
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        AllocateDataEvent,
        DeallocateDataEvent,
        BeginFenceEvent,
        BeginParallelForEvent
    >;
    using variant_t = typename recorder_listener_t::event_variant_t;
};

/**
 * @test Check that @ref Kokkos::Execution::make_memory_allocation works properly.
 *
 * The compute operation must complete properly, and the data is allocated only when the sender is sync-waited.
 */
TEST_F(TEST_CATEGORY(MemoryAllocationTest), compute_operation) {
    Kokkos::Execution::ExecutionSpaceContext<TEST_EXECUTION_SPACE> esc{exec};

    const Kokkos::View<int, Kokkos::SharedSpace> witness(Kokkos::view_alloc(exec, "witness"));

    const auto recorder_listener = std::make_shared<recorder_listener_t>();

    Kokkos::utils::callbacks::Manager::register_listener(recorder_listener);
    {
        auto sndr = exec::lifetime(
            [&esc, &witness]<typename T>(T& buffer) {
                static_assert(std::same_as<T, typename Kokkos::View<int, Kokkos::Serial>::mdspan_type>);
                return stdexec::schedule(esc.get_scheduler())
                     | Kokkos::Execution::parallel_for(42, Tests::Utils::Functors::SumIndices<T>{buffer})
                     | stdexec::then([witness, buffer]() noexcept {
                           witness() =
#if defined(MDSPAN_USE_BRACKET_OPERATOR) && MDSPAN_USE_BRACKET_OPERATOR == 1
                               buffer.operator[]();
#elif defined(MDSPAN_USE_BRACKET_OPERATOR) && MDSPAN_USE_BRACKET_OPERATOR == 0
                               buffer();
#else
#    error "MDSPAN_USE_BRACKET_OPERATOR must be defined."
#endif
                       });
            },
            Kokkos::Execution::make_memory_allocation<int, TEST_EXECUTION_SPACE>(Kokkos::view_alloc(exec, "data")));

        ASSERT_THAT(recorder_listener->recorded_events, ::testing::SizeIs(0));

        stdexec::sync_wait(std::move(sndr));

        ASSERT_THAT(
            recorder_listener->recorded_events,
            ContainsInOrder<variant_t>(
                MATCHER_FOR_ALLOCATE_DATA("data"),
                MATCHER_FOR_BEGIN_PFOR(
                    exec,
                    (Kokkos::Execution::Impl::display_string_of<Tests::Utils::Functors::SumIndices<
                         typename Kokkos::View<int, TEST_EXECUTION_SPACE>::mdspan_type
                     >>())),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_DEALLOCATE_DATA("data")));

        ASSERT_EQ(witness(), 861);
    }
    Kokkos::utils::callbacks::Manager::unregister_listener(recorder_listener.get());
}

} // namespace Tests
