#include "tests/kokkos_ext/execution_space/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Traits of @c Kokkos::Experimental::ExecutionSpaceContext
 * --------------------------------------------------------
 *
 * This group of tests check the traits of @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The tests can be found in @ref kokkos_ext/execution_space/test_Traits.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using ExecutionSpaceContextTest = impl::ExecutionSpaceContextTest<execution_space>;

/// @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext
///       satisfies the @c stdexec::scheduler concept.
///       Check that it has a valid schedule sender.
TEST_F(ExecutionSpaceContextTest, is_a_scheduler)
{
    static_assert(::stdexec::sender   <schedule_sender_t>);
    static_assert(::stdexec::scheduler<scheduler_t>);

    const context_t context{*exec};

    const stdexec::scheduler auto sch = context.get_scheduler();

    stdexec::sender auto sndr = stdexec::schedule(sch);
}

//! @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext has a custom domain.
TEST_F(ExecutionSpaceContextTest, has_custom_domain)
{
    static_assert(std::same_as<scheduler_domain_t, scheduler_t::Domain>);

    static_assert(std::same_as<stdexec::__early_domain_of_t<schedule_sender_t>, scheduler_t::Domain>);
}

} // namespace tests::kokkos_ext
