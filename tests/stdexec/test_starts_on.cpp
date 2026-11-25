#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::starts_on
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::starts_on.
 *
 * The tests can be found in @ref tests/stdexec/test_starts_on.cpp.
 */

namespace Tests {

template <stdexec::scheduler Schd, stdexec::sender Sndr>
using starts_on_result_t = decltype(stdexec::starts_on(std::declval<Schd>(), std::declval<Sndr>()));

//! @test @c stdexec::starts_on is a not dependent sender if children are not dependent senders.
consteval bool test_starts_on_is_not_a_dependent_sender_if_children_are_not_dependent() {
    using sndr_value_t = decltype(stdexec::just() | stdexec::then([]() noexcept { }));
    using sndr_value_and_error_t = decltype(stdexec::just() | stdexec::then([]() { }));

    static_assert(!stdexec::dependent_sender<sndr_value_t>);
    static_assert(!stdexec::dependent_sender<sndr_value_and_error_t>);

    static_assert(
        stdexec::get_completion_signatures<sndr_value_t>() == stdexec::completion_signatures<stdexec::set_value_t()>{});
    static_assert(
        stdexec::get_completion_signatures<sndr_value_and_error_t>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    //! The @c stdexec::inline_scheduler schedule sender is not a dependent sender.
    static_assert(!stdexec::dependent_sender<stdexec::schedule_result_t<stdexec::inline_scheduler>>);

    using inline_scheduler_value_t = starts_on_result_t<stdexec::inline_scheduler, sndr_value_t>;
    using inline_scheduler_value_and_error_t = starts_on_result_t<stdexec::inline_scheduler, sndr_value_and_error_t>;

    static_assert(!stdexec::dependent_sender<inline_scheduler_value_t>);
    static_assert(!stdexec::dependent_sender<inline_scheduler_value_and_error_t>);

    //! The @c exec::static_thread_pool schedule sender is a dependent sender.
    using pool_scheduler_t = decltype(std::declval<experimental::execution::static_thread_pool&>().get_scheduler());

    static_assert(stdexec::dependent_sender<stdexec::schedule_result_t<pool_scheduler_t>>);

    using pool_scheduler_value_t = starts_on_result_t<pool_scheduler_t, sndr_value_t>;
    using pool_scheduler_value_and_error_t = starts_on_result_t<pool_scheduler_t, sndr_value_and_error_t>;

    static_assert(stdexec::dependent_sender<pool_scheduler_value_t>);
    static_assert(stdexec::dependent_sender<pool_scheduler_value_and_error_t>);

    return true;
}

static_assert(test_starts_on_is_not_a_dependent_sender_if_children_are_not_dependent());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests
