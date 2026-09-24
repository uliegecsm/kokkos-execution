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

consteval bool test_starts_on_is_dependent_if_sch_sender_and_child_do_not_have_eptr_completion() {
    static_assert(!stdexec::__has_eptr_completion<decltype(stdexec::schedule(stdexec::inline_scheduler{}))>);

    using chain_t = decltype(stdexec::just());
    static_assert(!stdexec::__has_eptr_completion<chain_t>);

    static_assert(
        stdexec::dependent_sender<decltype(stdexec::starts_on(stdexec::inline_scheduler{}, std::declval<chain_t>()))>);

    return true;
}

/**
 * @test Check that for an eptr-free schedule sender and child, @c stdexec::starts_on is a dependent sender.
 *
 * See also:
 *  * https://github.com/NVIDIA/stdexec/blob/c4b51c48e0179f99439ccf23bf69d0b153185249/include/stdexec/__detail/__starts_on.hpp#L210
 *  * https://github.com/NVIDIA/stdexec/blob/c4b51c48e0179f99439ccf23bf69d0b153185249/include/stdexec/__detail/__sequence.hpp#L372-L376
 */
TEST(starts_on, is_dependent_if_sch_sender_and_child_do_not_have_eptr_completion) {
    static_assert(test_starts_on_is_dependent_if_sch_sender_and_child_do_not_have_eptr_completion());
}

consteval bool test_starts_on_may_be_non_dependent_if_child_has_eptr_completion() {
    using chain_t = decltype(stdexec::just() | stdexec::then([]() { }));
    static_assert(stdexec::__has_eptr_completion<chain_t>);

    static_assert(
        !stdexec::dependent_sender<decltype(stdexec::starts_on(stdexec::inline_scheduler{}, std::declval<chain_t>()))>);

    // The schedule sender of a static thread pool is dependent because it checks the environment
    // for a stop token.
    using pool_scheduler_t = decltype(std::declval<experimental::execution::static_thread_pool&>().get_scheduler());
    static_assert(!stdexec::__has_eptr_completion<decltype(stdexec::schedule(std::declval<pool_scheduler_t>()))>);
    static_assert(stdexec::dependent_sender<decltype(stdexec::schedule(std::declval<pool_scheduler_t>()))>);

    static_assert(stdexec::dependent_sender<decltype(stdexec::starts_on(
                      std::declval<pool_scheduler_t>(), std::declval<chain_t>()))>);

    return true;
}

/**
 * @test Check that for a schedule sender or a child with an eptr completion,
 *       @c stdexec::starts_on may be a non-dependent sender.
 */
TEST(starts_on, may_be_non_dependent_if_child_has_eptr_completion) {
    static_assert(test_starts_on_may_be_non_dependent_if_child_has_eptr_completion());
}

} // namespace Tests
