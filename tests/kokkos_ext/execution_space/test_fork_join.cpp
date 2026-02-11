#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/fork_join.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"

/**
 * @addtogroup unittests
 *
 * Use @c exec::fork_join with @c Kokkos::Experimental::ExecutionSpaceContext
 * --------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly works with @c exec::fork_join.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_fork_join.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class ForkJoinTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

//! @test Use @c exec::fork_join to produce a diamond-like topology.
TEST_F(ForkJoinTest, diamond) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    ::exec::static_thread_pool pool{4};
    const context_t esc{exec};

    auto chain =
        ::stdexec::schedule(pool.get_scheduler())
        | ::stdexec::then(::tests::utils::LoadCheckAddFunctor<int, false>{.prev = 0, .value = 4, .data = data.data()})
        | ::exec::fork_join(
            ::stdexec::continues_on(esc.get_scheduler()) | ::tests::stdexec::check_scheduler<scheduler_t>()
                | ADD_THEN_ATOMIC,
            ::stdexec::continues_on(pool.get_scheduler()) | ADD_THEN_ATOMIC)
        | ::stdexec::continues_on(esc.get_scheduler())
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 6, .value = 3, .data = data.data()})
        | ::stdexec::continues_on(::stdexec::inline_scheduler{})
        | ::stdexec::then(::tests::utils::LoadCheckAddFunctor<int, false>{.prev = 9, .value = 5, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 14);
}

/**
 * @test Use @c exec::fork_join after a @c stdexec::continues_on.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
 */
TEST_F(ForkJoinTest, continues_on) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    auto sndr =
        ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler())
        | ::exec::fork_join(
            ::stdexec::continues_on(esc.get_scheduler())
            | ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 0, .value = 3, .data = data.data()}));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { ::stdexec::sync_wait(std::move(sndr)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

} // namespace tests::kokkos_ext
