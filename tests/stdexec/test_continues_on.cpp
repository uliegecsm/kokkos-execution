#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c stdexec::continues_on
 * ----------------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::continues_on.
 *
 * The tests can be found in @ref tests/stdexec/test_continues_on.cpp.
 */

namespace Tests {

template <typename ViewType>
struct ContinuesOnSetErrorTest : public testing::Test { };

TYPED_TEST_SUITE_P(ContinuesOnSetErrorTest);

//! @test
TYPED_TEST_P(ContinuesOnSetErrorTest, set_error_t) {
    TypeParam scheduler_provider{};

    auto scheduler = scheduler_provider.get_scheduler();

    using scheduler_t = decltype(scheduler);

    auto sndr = stdexec::schedule(scheduler) | stdexec::then([]() { throw std::runtime_error{"error"}; })
              | stdexec::continues_on(stdexec::inline_scheduler{});

    using sndr_t = decltype(sndr);

    static_assert(
        stdexec::get_completion_signatures<sndr_t, stdexec::env<>>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    using start_scheduler_env_t = stdexec::prop<stdexec::get_start_scheduler_t, stdexec::run_loop::scheduler>;

    static_assert(stdexec::__sends<stdexec::set_value_t, sndr_t, stdexec::env<>>);
    static_assert(stdexec::__sends<stdexec::set_error_t, sndr_t, stdexec::env<>>);

    static_assert(stdexec::__sends<stdexec::set_value_t, sndr_t, start_scheduler_env_t>);
    static_assert(stdexec::__sends<stdexec::set_error_t, sndr_t, start_scheduler_env_t>);

    static_assert(!Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t, stdexec::env<>>);
    static_assert(!Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_error_t, stdexec::env<>>);

    //! Fails !
    static_assert(Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t, start_scheduler_env_t>);
    static_assert(Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_error_t, start_scheduler_env_t>);

    // static_assert(std::same_as<stdexec::__completion_scheduler_of_t<stdexec::set_value_t, sndr_t, stdexec::env<>>, scheduler_t>);
    // static_assert(std::same_as<stdexec::__completion_scheduler_of_t<stdexec::set_error_t, sndr_t, stdexec::env<>>, scheduler_t>);
}

REGISTER_TYPED_TEST_SUITE_P(ContinuesOnSetErrorTest, set_error_t);

using continues_on_set_error_types_t = ::testing::Types<
    experimental::execution::single_thread_context /*,
    experimental::execution::static_thread_pool*/
>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    ContinuesOnSetErrorTestInstantiation,
    ContinuesOnSetErrorTest,
    continues_on_set_error_types_t);

} // namespace Tests
