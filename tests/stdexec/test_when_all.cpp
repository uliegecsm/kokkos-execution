#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::when_all
 * ------------------------------
 *
 * This group of tests check the behavior of @c stdexec::when_all.
 *
 * The tests can be found in @ref tests/stdexec/test_when_all.cpp.
 */

namespace Tests {

using sndr_A_t = decltype(stdexec::when_all(
    stdexec::schedule(stdexec::inline_scheduler{}) | stdexec::then([]() { }),
    stdexec::just() | stdexec::then([]() { })));

using sndr_B_t = decltype(stdexec::when_all(
    stdexec::schedule(stdexec::inline_scheduler{}) | stdexec::then([]() { }),
    stdexec::schedule(std::declval<exec::single_thread_context>().get_scheduler()) | stdexec::then([]() { })));

//! @test Check the completion domain of @c stdexec::when_all.
consteval bool test_when_all_completion_domain() {
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_A_t>,
                  stdexec::indeterminate_domain<>
    >);

    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_A_t, stdexec::env<>>,
                  stdexec::default_domain
    >);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_B_t, stdexec::env<>>,
                  stdexec::default_domain
    >);

    return true;
}

static_assert(test_when_all_completion_domain());

//! @test Check when @c stdexec::when_all results in a dependent sender.
consteval bool test_when_all_dependent_sender() {
    static_assert(!stdexec::dependent_sender<sndr_A_t>);
    static_assert(stdexec::dependent_sender<sndr_B_t>);

    return true;
}

static_assert(test_when_all_dependent_sender());

//! @test A @c stdexec::when_all with no branch is equivalent to @c stdexec::just().
TEST(WhenAll, no_branch) {
    auto sndr = stdexec::when_all();

    static_assert(std::same_as<stdexec::tag_of_t<decltype(sndr)>, stdexec::just_t>);

    static_assert(!stdexec::dependent_sender<decltype(sndr)>);

    static_assert(
        stdexec::get_completion_signatures<decltype(sndr)>()
        == stdexec::completion_signatures<stdexec::set_value_t()>{});

    ASSERT_TRUE(stdexec::sync_wait(std::move(sndr)).has_value()); // NOLINT(performance-move-const-arg)
}

//! @test A @c stdexec::when_all(sndr) with a single branch is equivalent to @c auto(sndr).
TEST(WhenAll, single_branch) {
    auto sndr = stdexec::just(42);

    static_assert(std::same_as<decltype(stdexec::when_all(sndr)), decltype(sndr)>);
    static_assert(std::same_as<
                  decltype(stdexec::when_all(std::move(sndr))), // NOLINT(performance-move-const-arg)
                  decltype(sndr)
    >);

    auto&& w_a = stdexec::when_all(sndr);

    //! @c w_a is a new copy, not a reference to @c sndr.
    ASSERT_NE(std::addressof(w_a), std::addressof(sndr));

    sndr = stdexec::just(56);

    const auto result = stdexec::sync_wait(std::move(w_a)); // NOLINT(performance-move-const-arg)
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(std::get<0>(*result), 42);
    ASSERT_EQ(
        std::get<0>(*stdexec::sync_wait(stdexec::when_all(std::move(sndr)))), 56); // NOLINT(performance-move-const-arg)

    //! @c lvalues must be copyable; @c rvalues move.
    using just_not_copy_constructible_t = decltype(stdexec::just(std::make_unique<int>(0)));
    static_assert(!std::copy_constructible<just_not_copy_constructible_t>);
    static_assert(!std::invocable<const stdexec::when_all_t&, just_not_copy_constructible_t&>);
    static_assert(std::invocable<const stdexec::when_all_t&, just_not_copy_constructible_t&&>);
}

//! @test It is fine to pass branches with schedule senders only.
TEST(WhenAllTest, only_schedule_sender_branches) {
    exec::single_thread_context stc{};

    stdexec::sync_wait(
        stdexec::when_all(
            stdexec::schedule(stc.get_scheduler()),
            stdexec::schedule(stc.get_scheduler()),
            stdexec::schedule(stc.get_scheduler())));
}

} // namespace Tests
