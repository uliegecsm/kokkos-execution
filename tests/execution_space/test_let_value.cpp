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
    using recorder_listener_t =
        RecorderListener<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
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
                        return stdexec::just(view_of_5_t{Kokkos::view_alloc(exec, "scratch")});
                    });

    static_assert(
        Tests::Utils::has_completion_signatures<decltype(allocate), stdexec::__mset<stdexec::set_value_t(view_of_5_t)>>);

//! FIXME: https://github.com/kokkos/kokkos/blob/393d4165a6c3687e78abe5e1665853f1eabc386d/core/src/Kokkos_View.hpp#L697
#if defined(KOKKOS_COMPILER_CLANG) && defined(KOKKOS_ENABLE_CUDA)
    //! @c Kokkos::View is not nothrow movable, so the error channel is added.
    static_assert(Tests::Utils::has_completion_signatures<
                  decltype(allocate),
                  stdexec::__mset<stdexec::set_value_t(view_of_5_t), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::env<>
    >);
#else
    //! @c Kokkos::View is nothrow movable, so the error channel is not added.
    static_assert(Tests::Utils::has_completion_signatures<
                  decltype(allocate),
                  stdexec::__mset<stdexec::set_value_t(view_of_5_t)>,
                  stdexec::env<>
    >);
#endif

    //! Use the scratch view to make some meaningful computation.
    auto run = std::move(allocate) // NOLINT(performance-move-const-arg)
             | stdexec::let_value(
                   [&esc, &data, &stc](
                       Kokkos::utils::concepts::ViewOfRank<1> auto const & scratch) noexcept -> stdexec::sender auto {
                       return stdexec::schedule(esc.get_scheduler())
                            | stdexec::bulk(
                                  stdexec::par,
                                  5,
                                  KOKKOS_LAMBDA<std::integral T>(const T index) {
                                      scratch(index) = index;
                                      Kokkos::atomic_add(data.data(), scratch(index));
                                  })
                            | stdexec::continues_on(stc.get_scheduler())
                            | stdexec::then([scratch]() mutable -> Kokkos::utils::concepts::ViewOfRank<1> auto {
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
               | stdexec::let_value(
                     [&esc, &data](
                         Kokkos::utils::concepts::ViewOfRank<1> auto const & scratch) noexcept -> stdexec::sender auto {
                         return stdexec::schedule(esc.get_scheduler())
                              | stdexec::bulk(
                                    stdexec::par, 5, KOKKOS_LAMBDA<std::integral T>(const T index) {
                                        Kokkos::atomic_add(data.data(), scratch(index));
                                    });
                     });

    static_assert(Tests::Utils::has_completion_signatures<
                  decltype(check),
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::env<>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([check = std::move(check)]() mutable { // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(check));                          // NOLINT(performance-move-const-arg)
        }),
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
