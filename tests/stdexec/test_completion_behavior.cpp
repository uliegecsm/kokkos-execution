#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Completion behavior
 * -------------------
 *
 * This group of tests check completion behavior and related utilities.
 *
 * The tests can be found in @ref tests/stdexec/test_completion_behavior.cpp.
 */

namespace Tests {

//! @test The completion behavior of @c exec::single_thread_context is @c stdexec::__completion_behavior::__asynchronous.
consteval bool test_single_thread_context_completion_behavior_is_asynchronous() {
    using sndr_t = std::invoke_result_t<
        stdexec::then_t,
        stdexec::schedule_result_t<decltype(std::declval<exec::single_thread_context>().get_scheduler())>,
        Tests::Utils::Functors::NoOp<true, false, false>
    >;

    using start_scheduler_env_t = stdexec::prop<stdexec::get_start_scheduler_t, stdexec::run_loop::scheduler>;

    static_assert(stdexec::__sends<stdexec::set_value_t, sndr_t, start_scheduler_env_t>);
    static_assert(stdexec::__sends<stdexec::set_error_t, sndr_t, start_scheduler_env_t>);

    static_assert(
        !stdexec::__completes_where_it_starts<stdexec::set_value_t, stdexec::env_of_t<sndr_t>, start_scheduler_env_t>);
    static_assert(
        !stdexec::__completes_where_it_starts<stdexec::set_error_t, stdexec::env_of_t<sndr_t>, start_scheduler_env_t>);

    static_assert(
        stdexec::__get_completion_behavior<stdexec::set_value_t, sndr_t, start_scheduler_env_t>()
        == stdexec::__completion_behavior::__asynchronous);
    static_assert(
        stdexec::__get_completion_behavior<stdexec::set_error_t, sndr_t, start_scheduler_env_t>()
        == stdexec::__completion_behavior::__asynchronous);

    return true;
}

static_assert(test_single_thread_context_completion_behavior_is_asynchronous());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests
