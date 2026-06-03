#include "gmock/gmock.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"

/**
 * @addtogroup unittests
 *
 * Completion signatures
 * ---------------------
 *
 * This group of tests check our helpers for computing completion signatures from @ref kokkos-execution/impl/completion_signatures.hpp.
 *
 * The tests can be found in @ref tests/impl/test_completion_signatures.cpp.
 */

namespace Tests::Impl {

using sndr_t =
    decltype(stdexec::schedule(std::declval<experimental::execution::single_thread_context>().get_scheduler()) | stdexec::then([]() noexcept {
             }));

//! @test Check return type of @ref Kokkos::Execution::Impl::completion_signatures_add_t without added completion signature and an empty environment.
consteval bool test_add_nothing_empty_env() {
    static_assert(
        stdexec::__mset_eq<
            stdexec::__mset<stdexec::set_value_t()>,
            Kokkos::Execution::Impl::completion_signatures_add_t<sndr_t, stdexec::completion_signatures<>, stdexec::env<>>
        >);
    return true;
}
static_assert(test_add_nothing_empty_env());

//! @test Check return type of @ref Kokkos::Execution::Impl::completion_signatures_add_t with an added error completion signature and an empty environment.
consteval bool test_add_error_empty_env() {
    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(float)>,
                  Kokkos::Execution::Impl::completion_signatures_add_t<
                      sndr_t,
                      stdexec::completion_signatures<stdexec::set_error_t(float)>,
                      stdexec::env<>
                  >
    >);
    return true;
}
static_assert(test_add_error_empty_env());

using env_with_stop_token_t = stdexec::prop<stdexec::get_stop_token_t, stdexec::inplace_stop_token>;

//! @test Check return type of @ref Kokkos::Execution::Impl::completion_signatures_add_t without added completion signature and @ref env_with_stop_token_t.
consteval bool test_add_nothing_stop_env() {
    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_stopped_t()>,
                  Kokkos::Execution::Impl::completion_signatures_add_t<
                      sndr_t,
                      stdexec::completion_signatures<>,
                      env_with_stop_token_t
                  >
    >);
    return true;
}
static_assert(test_add_nothing_stop_env());

//! @test Check return type of @ref Kokkos::Execution::Impl::completion_signatures_add_t with an added error completion signature and @ref env_with_stop_token_t.
consteval bool test_add_error_stop_env() {
    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_stopped_t(), stdexec::set_error_t(float)>,
                  Kokkos::Execution::Impl::completion_signatures_add_t<
                      sndr_t,
                      stdexec::completion_signatures<stdexec::set_error_t(float)>,
                      env_with_stop_token_t
                  >
    >);

    return true;
}
static_assert(test_add_error_stop_env());

class CompletionSignaturesTest : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE> { };

/**
 * @test Check the propagation and addition of the value/error/stopped channels.
 *
 * This test implicitly exercices @ref KOKKOS_EXECUTION_COMPL_SIGS_ADD and @ref KOKKOS_EXECUTION_COMPL_SIGS_KEEP.
 */
TEST_F(CompletionSignaturesTest, parallel_for) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    experimental::execution::single_thread_context stc{};

    const context_t esc{exec};

    stdexec::inplace_stop_source source;

    auto stc_then_continues_on_esc_sndr = stdexec::schedule(stc.get_scheduler()) | stdexec::then([]() noexcept { })
                                        | stdexec::continues_on(esc.get_scheduler());

    //! The stopped channel of the @c experimental::execution::single_thread_context is properly propagated.
    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_stopped_t()>,
                  stdexec::__completion_signatures_of_t<decltype(stc_then_continues_on_esc_sndr), env_with_stop_token_t>
    >);

    auto stc_then_continues_on_esc_then_sndr =
        std::move(stc_then_continues_on_esc_sndr) // NOLINT(performance-move-const-arg)
        | stdexec::then(Tests::Utils::Functors::Increment<view_s_t, true, false>{.data = data});

    //! The stopped channel is propagated, and the error channel is added.
    static_assert(
        stdexec::__mset_eq<
            stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr), stdexec::set_stopped_t()>,
            stdexec::__completion_signatures_of_t<decltype(stc_then_continues_on_esc_then_sndr), env_with_stop_token_t>
        >);

    auto stc_then_continues_on_esc_then_then_sndr =
        std::move(stc_then_continues_on_esc_then_sndr)
        | stdexec::then(Tests::Utils::Functors::Increment<view_s_t, true, false>{.data = data});

    //! The error channel was already added, it is added only once.
    static_assert(
        stdexec::__mset_eq<
            stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr), stdexec::set_stopped_t()>,
            stdexec::__completion_signatures_of_t<
                decltype(stc_then_continues_on_esc_then_then_sndr),
                env_with_stop_token_t
            >
        >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_FALSE(source.stop_requested());

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(stc_then_continues_on_esc_then_then_sndr)

    const auto res_A = stdexec::sync_wait(
        stc_then_continues_on_esc_then_then_sndr
        | stdexec::write_env(stdexec::prop{stdexec::get_stop_token, source.get_token()}));

    ASSERT_TRUE(res_A.has_value());

    ASSERT_EQ(data(), 2);

    source.request_stop();

    const auto res_B = stdexec::sync_wait(
        std::move(stc_then_continues_on_esc_then_then_sndr)
        | stdexec::write_env(stdexec::prop{stdexec::get_stop_token, source.get_token()}));

    ASSERT_FALSE(res_B.has_value());

    ASSERT_EQ(data(), 2);
}

} // namespace Tests::Impl
