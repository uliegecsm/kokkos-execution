#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Behavior of @c let_value with @c Kokkos::Experimental::ExecutionSpaceContext
 * ----------------------------------------------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::let_value when used with the
 * @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_let_value.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class LetValueTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
    using variant_t = std::variant<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
};

//! @test Use the value channel and @c stdexec::let_value to keep a "scratch" @c Kokkos view alive during the computations.
TEST_F(LetValueTest, scoped_allocation) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    ::exec::static_thread_pool pool{1};

    using view_of_5_t = Kokkos::View<value_t[5], typename execution_space::memory_space>;

    //! Allocate a @c Kokkos view only when the sender is running, put it in the value channel.
    auto allocate = ::stdexec::schedule(esc.get_scheduler())
                  | ::stdexec::let_value([this]() noexcept -> ::stdexec::sender auto {
                        return ::stdexec::just(view_of_5_t{Kokkos::view_alloc(exec, "scratch")});
                    });

    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(allocate),
                  ::stdexec::__mset<::stdexec::set_value_t(view_of_5_t)>
    >);

    //! @c Kokkos view is not nothrow movable, so the error channel is added.
    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(allocate),
                  ::stdexec::__mset<::stdexec::set_value_t(view_of_5_t), ::stdexec::set_error_t(std::exception_ptr)>,
                  ::stdexec::env<>
    >);

    //! Use the scratch view to make some meaningful computation.
    auto run = std::move(allocate) // NOLINT(performance-move-const-arg)
             | ::stdexec::let_value(
                   [&esc, &data, &pool](
                       Kokkos::utils::concepts::ViewOfRank<1> auto const & scratch) noexcept -> ::stdexec::sender auto {
                       return ::stdexec::schedule(esc.get_scheduler())
                            | ::stdexec::bulk(
                                  ::stdexec::par,
                                  5,
                                  KOKKOS_LAMBDA<std::integral T>(const T index) {
                                      scratch(index) = index;
                                      Kokkos::atomic_add(data.data(), scratch(index));
                                  })
                            | ::stdexec::continues_on(pool.get_scheduler())
                            | ::stdexec::then([scratch]() mutable -> Kokkos::utils::concepts::ViewOfRank<1> auto {
                                  return std::move(scratch);
                              });
                   });

    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(run),
                  ::stdexec::__mset<::stdexec::set_value_t(view_of_5_t), ::stdexec::set_error_t(std::exception_ptr)>,
                  ::stdexec::env<>
    >);

    //! Check the result of the computation.
    auto check = std::move(run) // NOLINT(performance-move-const-arg)
               | ::stdexec::let_value(
                     [&esc, &data](Kokkos::utils::concepts::ViewOfRank<1> auto const & scratch) noexcept
                         -> ::stdexec::sender auto {
                         return ::stdexec::schedule(esc.get_scheduler())
                              | ::stdexec::bulk(
                                    ::stdexec::par, 5, KOKKOS_LAMBDA<std::integral T>(const T index) {
                                        Kokkos::atomic_add(data.data(), scratch(index));
                                    });
                     });

    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(check),
                  ::stdexec::__mset<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>,
                  ::stdexec::env<>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([check = std::move(check)]() mutable { // NOLINT(performance-move-const-arg)
            ::stdexec::sync_wait(std::move(check));                        // NOLINT(performance-move-const-arg)
        }),
        ContainsInOrder<variant_t>(
            Kokkos::utils::callbacks::AAllocateDataEvent(
                ::testing::Field(
                    &Kokkos::utils::callbacks::AllocateDataEvent::alloc,
                    ::testing::Field(&Kokkos::utils::callbacks::AllocDescriptor::name, ::testing::StrEq("scratch")))),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")),
            Kokkos::utils::callbacks::ADeallocateDataEvent(
                ::testing::Field(
                    &Kokkos::utils::callbacks::DeallocateDataEvent::alloc,
                    ::testing::Field(&Kokkos::utils::callbacks::AllocDescriptor::name, ::testing::StrEq("scratch"))))));

    ASSERT_EQ(data(), 2 * (0 + 1 + 2 + 3 + 4));
}

} // namespace tests::kokkos_ext
