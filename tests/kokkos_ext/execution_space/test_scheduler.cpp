#include "gtest/gtest.h"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"

/**
 * @addtogroup unittests
 *
 * Traits of the scheduler of @c Kokkos::Experimental::ExecutionSpaceContext
 * -------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::details::execution_space::Scheduler is a proper scheduler.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_scheduler.cpp.
 *
 * References:
 *  * https://eel.is/c++draft/exec.sched
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using execution_space_context_t = Kokkos::Experimental::ExecutionSpaceContext<execution_space>;
using execution_space_scheduler_t = Kokkos::Experimental::details::execution_space::Scheduler<execution_space>;
using execution_space_scheduler_env_t = Kokkos::Experimental::details::execution_space::SchedulerEnv<execution_space>;
using execution_space_schedule_sender_t = typename execution_space_scheduler_t::Sender;

//! @test @ref Kokkos::Experimental::details::execution_space::Scheduler models the @c stdexec::scheduler concept.
constexpr bool test_scheduler_concept() {
    static_assert(stdexec::scheduler<execution_space_scheduler_t>);

    /**
     * According to https://eel.is/c++draft/exec.sched#1, a valid scheduler must have a @c scheduler_concept
     * alias.
     * However, as of https://github.com/NVIDIA/stdexec/blob/0e9983599d0c95fca3fd11baa02564eb53fb14f6/include/stdexec/__detail/__schedulers.hpp#L74,
     * it is not checked by @c stdexec::scheduler.
     *
     * Related to https://github.com/NVIDIA/stdexec/issues/1406.
     */
    static_assert(std::derived_from<typename execution_space_scheduler_t::scheduler_concept, stdexec::scheduler_t>);

    //! According to https://eel.is/c++draft/exec.sched#1, a @c schedule invocation must return a sender.
    static_assert(stdexec::sender<decltype(stdexec::schedule(std::declval<const execution_space_scheduler_t&>()))>);

    return true;
}
static_assert(test_scheduler_concept());

/**
 * @test Check that querying the completion scheduler from a schedule sender of a scheduler returns the scheduler.
 * 
 * See https://eel.is/c++draft/exec.sched#5.
 */
TEST(Scheduler, round_trip_property) {
    const execution_space_context_t ctx{execution_space{}};
    const execution_space_scheduler_t sch = ctx.get_scheduler();
    ASSERT_EQ(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(stdexec::schedule(sch))), sch);
}

/**
 * @test Check that the @c schedule method of @ref Kokkos::Experimental::details::execution_space::Scheduler returns
 *       @ref Kokkos::Experimental::details::execution_space::Scheduler::Sender.
 */
constexpr bool test_scheduler_schedule() {
    static_assert(std::same_as<
                  decltype(stdexec::schedule(std::declval<const execution_space_scheduler_t&>())),
                  execution_space_schedule_sender_t
    >);

    return true;
}

/**
 * @test Check that the @c stdexec::get_env query on @ref Kokkos::Experimental::details::execution_space::Scheduler::Sender returns
 *       @ref Kokkos::Experimental::details::execution_space::SchedulerEnv.
 */
constexpr bool test_schedule_sender_env() {
    static_assert(
        std::same_as<stdexec::env_of_t<execution_space_schedule_sender_t>, const execution_space_scheduler_env_t&>);

    return true;
}
static_assert(test_schedule_sender_env());

//! @test Check @ref Kokkos::Experimental::details::execution_space::Scheduler queries.
constexpr bool test_scheduler_queries() {
    static_assert(std::same_as<
                  stdexec::__query_result_t<
                      execution_space_scheduler_t,
                      stdexec::get_completion_domain_t<stdexec::set_value_t>
                  >,
                  Kokkos::Experimental::details::execution_space::Domain
    >);

    static_assert(std::same_as<
                  stdexec::__query_result_t<
                      execution_space_scheduler_t,
                      stdexec::get_completion_scheduler_t<stdexec::set_value_t>
                  >,
                  execution_space_scheduler_t
    >);

    return true;
}
static_assert(test_scheduler_queries());

//! @test Check queries of @ref Kokkos::Experimental::details::execution_space::SchedulerEnv.
constexpr bool test_scheduler_env_queries() {
    static_assert(std::same_as<
                  stdexec::__query_result_t<
                      execution_space_scheduler_env_t,
                      stdexec::get_completion_scheduler_t<stdexec::set_value_t>
                  >,
                  execution_space_scheduler_t
    >);

    static_assert(std::same_as<
                  stdexec::__query_result_t<
                      execution_space_scheduler_env_t,
                      stdexec::get_completion_domain_t<stdexec::set_value_t>
                  >,
                  Kokkos::Experimental::details::execution_space::Domain
    >);

    return true;
}
static_assert(test_scheduler_env_queries());

} // namespace tests::kokkos_ext
