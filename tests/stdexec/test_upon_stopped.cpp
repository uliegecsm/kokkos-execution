#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Check how @c stdexec::upon_stopped works
 * ----------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::upon_stopped.
 *
 * The tests can be found in @ref tests/stdexec/test_upon_stopped.cpp.
 */

namespace Tests {

//! An environment queryable for @c stdexec::get_stop_token_t.
using env_with_stop_token_t = stdexec::env<stdexec::prop<stdexec::get_stop_token_t, stdexec::inplace_stop_token>>;

//! @test @c stdexec::set_stopped_t() is advertised only if the environment has a stoppable token.
template <typename Sndr>
consteval bool test_stopped_advertised_iff_stop_token() {
    static_assert(!stdexec::sends_stopped<Sndr, stdexec::env<>>);
    static_assert(stdexec::sends_stopped<Sndr, env_with_stop_token_t>);

    return true;
}

//! @test A stop requested before start skips the work and reaches @c stdexec::upon_stopped.
template <bool Stop, stdexec::sender SchdSndr>
void test_stop_before_start(SchdSndr&& schd_sndr) {
    stdexec::inplace_stop_source iss; // NOLINT(misc-const-correctness)

    if constexpr (Stop)
        iss.request_stop();

    unsigned int value = 0;

    auto sndr_then = std::forward<SchdSndr>(schd_sndr) | stdexec::then([&value]() noexcept {
                         value += 12;
                         return 666;
                     });

    static_assert(test_stopped_advertised_iff_stop_token<decltype(sndr_then)>());

    static_assert(
        stdexec::get_completion_signatures<decltype(sndr_then), stdexec::env<>>()
        == stdexec::completion_signatures<stdexec::set_value_t(int)>{});

    auto sndr_upon_stopped = std::move(sndr_then) | stdexec::upon_stopped([&value]() noexcept {
                                 value += 56;
                                 return 42;
                             });

    static_assert(
        stdexec::get_completion_signatures<decltype(sndr_then), stdexec::env<>>()
        == stdexec::completion_signatures<stdexec::set_value_t(int)>{});

    //! @c stdexec::upon_stopped absorbs the @c stdexec::set_stopped_t channel.
    static_assert(STDEXEC::__never_sends<stdexec::set_stopped_t, decltype(sndr_upon_stopped), env_with_stop_token_t>);

    auto sndr = std::move(sndr_upon_stopped)
              | stdexec::write_env(stdexec::prop{stdexec::get_stop_token, iss.get_token()});

    const auto res = stdexec::sync_wait(std::move(sndr));

    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(std::get<0>(*res), Stop ? 42 : 666);
    ASSERT_EQ(value, Stop ? 56 : 12);
}

template <typename T>
requires requires(T& arg) {
    { arg.get_scheduler() } -> stdexec::scheduler;
}
class UponStoppedTest : public testing::Test { };

TYPED_TEST_SUITE_P(UponStoppedTest);

TYPED_TEST_P(UponStoppedTest, stop_before_start) {
    static_assert(test_stopped_advertised_iff_stop_token<decltype(stdexec::schedule(
                      std::declval<TypeParam>().get_scheduler()))>());

    TypeParam scheduler_provider{};

    test_stop_before_start<true>(stdexec::schedule(scheduler_provider.get_scheduler()));
    test_stop_before_start<false>(stdexec::schedule(scheduler_provider.get_scheduler()));
}

REGISTER_TYPED_TEST_SUITE_P(UponStoppedTest, stop_before_start);

using UponStoppedTestTypes = testing::Types<exec::static_thread_pool, exec::single_thread_context>;

INSTANTIATE_TYPED_TEST_SUITE_P(UponStoppedTestInstantiation, UponStoppedTest, UponStoppedTestTypes);

} // namespace Tests
