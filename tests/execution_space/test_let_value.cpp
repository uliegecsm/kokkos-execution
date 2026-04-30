#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/concepts/View.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Behavior of @c let_value with @c Kokkos::Execution::ExecutionSpaceContext
 * -------------------------------------------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::let_value when used with the
 * @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * The tests can be found in @ref tests/execution_space/test_let_value.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class LetValueTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        BeginFenceEvent,
        BeginParallelForEvent,
        AllocateDataEvent,
        DeallocateDataEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
    using variant_t = typename recorder_listener_t::event_variant_t;
};

//! @test Use the value channel and @c stdexec::let_value to keep a "scratch" @c Kokkos view alive during the computations.
TEST_F(LetValueTest, scoped_allocation) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    experimental::execution::single_thread_context stc{};

    using view_of_5_t = Kokkos::View<value_t[5], typename TEST_EXECUTION_SPACE::memory_space>;

    //! Allocate a @c Kokkos view only when the sender is running, put it in the value channel.
    auto allocate = stdexec::schedule(esc.get_scheduler())
                  | stdexec::let_value([this]() noexcept -> stdexec::sender auto {
                        return stdexec::just(
                            view_of_5_t{Kokkos::view_alloc(exec, "scratch", Kokkos::WithoutInitializing)});
                    });

    static_assert(stdexec::dependent_sender<decltype(allocate)>);

    //! @c Kokkos::View is nothrow movable, so the error channel is not added.
    static_assert(Tests::Utils::has_completion_signatures<
                  decltype(allocate),
                  stdexec::__mset<stdexec::set_value_t(view_of_5_t)>,
                  stdexec::env<>
    >);

    //! Use the scratch view to make some meaningful computation.
    auto run = std::move(allocate) // NOLINT(performance-move-const-arg)
             /**
              * The operation state of @c stdexec::let_value stores the arguments in a decayed tuple
              * and passes references to them when invoking the lambda.
              *
              * - https://github.com/NVIDIA/stdexec/blob/f662722de0de4b4795c56fe8546ecf9412ec5a3f/include/stdexec/__detail/__let.hpp#L315
              * - https://github.com/NVIDIA/stdexec/blob/f662722de0de4b4795c56fe8546ecf9412ec5a3f/include/stdexec/__detail/__let.hpp#L355
              * - https://github.com/NVIDIA/stdexec/blob/f662722de0de4b4795c56fe8546ecf9412ec5a3f/include/stdexec/__detail/__let.hpp#L358
              */
             | stdexec::let_value([&esc, &data, &stc](auto&& scratch) noexcept -> stdexec::sender auto {
                   static_assert(std::same_as<decltype(scratch), view_of_5_t&>);
                   EXPECT_EQ(scratch.use_count(), 1);
                   return stdexec::schedule(esc.get_scheduler())
                        | stdexec::bulk(
                              stdexec::par,
                              5,
                              KOKKOS_LAMBDA<std::integral T>(const T index) {
                                  /**
                                   * The use count includes the views in the @c stdexec::let_value operation state and the @c stdexec::bulk closure.
                                   *
                                   * There is also a copy in the @c Kokkos parallel region, but it is not included in the use count
                                   * because of @c Kokkos::parallel_for uses @c Kokkos::Impl::construct_with_shared_allocation_tracking_disabled.
                                   */
                                  KOKKOS_IF_ON_HOST(EXPECT_EQ(scratch.use_count(), 2);)
                                  scratch(index) = index;
                                  Kokkos::atomic_add(data.data(), scratch(index));
                              })
                        | stdexec::continues_on(stc.get_scheduler())
                        | stdexec::then([&scratch]() mutable -> Kokkos::utils::concepts::ViewOfRank<1> auto {
                              //! The use count includes the views in the @c stdexec::let_value operation state and the @c stdexec::bulk closure.
                              EXPECT_EQ(scratch.use_count(), 2);
                              return std::move(scratch);
                          });
               });

    static_assert(Tests::Utils::has_completion_signatures<
                  decltype(run),
                  stdexec::__mset<stdexec::set_value_t(view_of_5_t), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::env<>
    >);

    //! Check the result of the computation.
    auto check = std::move(run) // NOLINT(performance-move-const-arg)
               | stdexec::let_value([&esc, &data](auto&& scratch) noexcept -> stdexec::sender auto {
                     static_assert(std::same_as<decltype(scratch), view_of_5_t&>);
                     /**
                      * The implementation of @c stdexec::let_value uses a single variant to store the predecessor and successor operation states.
                      *
                      * The predecessor operation state is still alive when the lambda is invoked to construct the successor sender. The predecessor
                      * operation state is destroyed only when the successor sender is connected and assigned to the variant.
                      *
                      * - https://github.com/NVIDIA/stdexec/blob/f662722de0de4b4795c56fe8546ecf9412ec5a3f/include/stdexec/__detail/__let.hpp#L357-L365
                      * - https://github.com/NVIDIA/stdexec/blob/f662722de0de4b4795c56fe8546ecf9412ec5a3f/include/stdexec/__detail/__let.hpp#L382-L384
                      *
                      * The use count includes the views in the "run" @c stdexec::bulk closure and the "check" @c stdexec::let_value operation state.
                      *
                      * The "run" @c stdexec::then lambda moved the view held by the "run" @c stdexec::let_value operation state, so it no longer
                      * contributes to the use count at this point.
                      */
                     EXPECT_EQ(scratch.use_count(), 2);
                     return stdexec::schedule(esc.get_scheduler())
                          | stdexec::bulk(
                                stdexec::par, 5, KOKKOS_LAMBDA<std::integral T>(const T index) {
                                    Kokkos::atomic_add(data.data(), scratch(index));
                                    /**
                                     * The use count includes the views in the "check" @c stdexec::let_value operation state and the @c stdexec::bulk closure
                                     *
                                     * There is also a copy in the @c Kokkos parallel region, but it is not included in the use count
                                     * because of @c Kokkos::parallel_for uses @c Kokkos::Impl::construct_with_shared_allocation_tracking_disabled.
                                     */
                                    KOKKOS_IF_ON_HOST(EXPECT_EQ(scratch.use_count(), 2);)
                                });
                 });

    static_assert(Tests::Utils::has_completion_signatures<
                  decltype(check),
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::env<>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(check)

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(check)), // NOLINT(performance-move-const-arg)
        ContainsInOrder<variant_t>(
            Kokkos::utils::callbacks::AAllocateDataEvent(
                testing::Field(
                    &Kokkos::utils::callbacks::AllocateDataEvent::alloc,
                    testing::Field(&Kokkos::utils::callbacks::AllocDescriptor::name, testing::StrEq("scratch")))),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")),
            Kokkos::utils::callbacks::ADeallocateDataEvent(
                testing::Field(
                    &Kokkos::utils::callbacks::DeallocateDataEvent::alloc,
                    testing::Field(&Kokkos::utils::callbacks::AllocDescriptor::name, testing::StrEq("scratch"))))));

    ASSERT_EQ(data(), 2 * (0 + 1 + 2 + 3 + 4));
}

} // namespace Tests::ExecutionSpaceImpl
