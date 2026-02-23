#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "exec/any_sender_of.hpp"
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
 * Type erased senders and @c Kokkos::Experimental::ExecutionSpaceContext
 * ----------------------------------------------------------------------
 *
 * This group of tests check that type erased senders are usable with
 * @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The key design point is that the type erased sender no longer knows the concrete sender type,
 * so the only way to support environment‑based customization is via a fixed set of queries that
 * the erasure wrapper promises to forward. See:
 *  - https://github.com/NVIDIA/stdexec/blob/fa05bc3c93d85c22e8fd987c3b96412a9980f183/include/exec/any_sender_of.hpp#L1310
 *  - https://github.com/NVIDIA/stdexec/blob/fa05bc3c93d85c22e8fd987c3b96412a9980f183/include/exec/any_sender_of.hpp#L1353
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_any_sender.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class AnySenderTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

/**
 * @test In order to get the same synchronization behavior as if using fully typed senders,
 *       type erased senders must advertise the completion domain and scheduler and type erased receivers
 *       must advertise their @ref Kokkos::Experimental::details::execution_space::get_exec_t query.
 */
TEST_F(AnySenderTest, then) {
    using any_sender_t = ::exec::any_receiver_ref<
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>,
        Kokkos::Experimental::details::execution_space::get_exec
            .signature<Kokkos::Experimental::details::execution_space::ExecutionSpaceRef<execution_space>() noexcept>
    >::
        template any_sender<
            ::stdexec::get_completion_scheduler<::stdexec::set_value_t>.signature<Kokkos::Experimental::details::execution_space::Scheduler<execution_space>() noexcept>,
            ::stdexec::get_completion_domain<::stdexec::set_value_t>.signature<Kokkos::Experimental::details::execution_space::Domain() noexcept>
        >;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    any_sender_t one = ::stdexec::schedule(esc.get_scheduler()) | ADD_THEN | ADD_THEN;

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(one)>,
                  Kokkos::Experimental::details::execution_space::Domain
    >);
    static_assert(std::same_as<
                  ::stdexec::__completion_scheduler_of_t<::stdexec::set_value_t, decltype(one)>,
                  typename AnySenderTest::scheduler_t
    >);

    auto continues_on = std::move(one) | ::stdexec::continues_on(esc.get_scheduler());

    static_assert(std::same_as<
                  ::stdexec::__demangle_t<decltype(continues_on)>,
                  ::tests::stdexec::basic_sender<
                      ::stdexec::continues_on_t,
                      typename AnySenderTest::scheduler_t,
                      ::tests::stdexec::basic_sender<::stdexec::schedule_from_t, ::stdexec::__, any_sender_t>
                  >
    >);

    any_sender_t two = std::move(continues_on) | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([two = std::move(two)]() mutable { ::stdexec::sync_wait(std::move(two)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

} // namespace tests::kokkos_ext
