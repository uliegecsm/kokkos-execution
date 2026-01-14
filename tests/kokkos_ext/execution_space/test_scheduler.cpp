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

using execution_space_scheduler_t =
    decltype(std::declval<const Kokkos::Experimental::ExecutionSpaceContext<execution_space>&>().get_scheduler());
using execution_space_env_t = Kokkos::Experimental::details::execution_space::SchedulerEnv<execution_space>;

//! @test @ref Kokkos::Experimental::details::execution_space::Scheduler verifies @c stdexec::scheduler.
constexpr bool test_is_a_scheduler() {
    static_assert(stdexec::scheduler<execution_space_scheduler_t>);

    /**
     * According to https://eel.is/c++draft/exec.sched#1, a valid scheduler must have a @c scheduler_concept
     * alias.
     * However, as of https://github.com/NVIDIA/stdexec/blob/0e9983599d0c95fca3fd11baa02564eb53fb14f6/include/stdexec/__detail/__schedulers.hpp#L74,
     * it is not checked by @c stdexec::scheduler.
     *
     * Related to https://github.com/NVIDIA/stdexec/issues/1406.
     */
    return std::derived_from<typename execution_space_scheduler_t::scheduler_concept, stdexec::scheduler_t>;
}
static_assert(test_is_a_scheduler());

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

/**
 * @test Check that the @c stdexec::get_env query on @ref Kokkos::Experimental::details::execution_space::Scheduler::Sender returns
 *       @ref Kokkos::Experimental::details::execution_space::SchedulerEnv.
 */
constexpr bool test_scheduler_sender_get_env() {
    static_assert(
        std::same_as<stdexec::env_of_t<typename execution_space_scheduler_t::Sender>, const execution_space_env_t&>);

    return true;
}
static_assert(test_scheduler_sender_get_env());

//! @test Check queries of @ref Kokkos::Experimental::details::execution_space::SchedulerEnv.
constexpr bool test_environment_queries() {
    static_assert(
        std::same_as<
            stdexec::__query_result_t<execution_space_env_t, stdexec::get_completion_scheduler_t<stdexec::set_value_t>>,
            execution_space_scheduler_t
        >);

    static_assert(
        std::same_as<
            stdexec::__query_result_t<execution_space_env_t, stdexec::get_completion_domain_t<stdexec::set_value_t>>,
            Kokkos::Experimental::details::execution_space::Domain
        >);

    return true;
}
static_assert(test_environment_queries());

} // namespace tests::kokkos_ext
