#include "gtest/gtest.h"

#include "tests/kokkos_ext/execution_space/Helpers.hpp"

#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c continues_on by @c Kokkos::Experimental::ExecutionSpaceContext
 * ----------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c continues_on.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_continues_on.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

class ContinuesOnTest : public impl::ExecutionSpaceContextTest<execution_space>
{};

//! @test Similar to @ref tests::stdexec::adaptors::ContinuesOnTest_no_schedule_sender_continues_on_Test.
TEST_F(ContinuesOnTest, completing_domain)
{
    const context_t esc{exec};

    ::stdexec::sender auto sndr = ::stdexec::just(42) | ::stdexec::continues_on(esc.get_scheduler());

    int placeholder = 0;

    auto opstate = ::stdexec::connect(std::move(sndr), ::tests::stdexec::value_receiver<int>{.value = std::addressof(placeholder)});

    ::stdexec::start(opstate);

    ASSERT_EQ(placeholder, 42);

    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<decltype(sndr)>>,
        ::stdexec::default_domain
    >);

    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, decltype(sndr)>,
        Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<execution_space>::Domain
    >);

    //! @todo This should be working once properly customized.
    static_assert(!::tests::stdexec::has_completion_scheduler_for<decltype(sndr), ::stdexec::set_value_t>);
}

} // namespace tests::kokkos_ext
