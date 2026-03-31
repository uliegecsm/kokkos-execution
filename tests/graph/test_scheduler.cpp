#include "gtest/gtest.h"

#include "tests/utils/check_scheduler.hpp"
#include "tests/utils/graph_context.hpp"

/**
 * @addtogroup unittests
 *
 * Traits of the scheduler of @c Kokkos::Execution::GraphContext
 * -------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphImpl::Scheduler is a proper scheduler.
 *
 * The tests can be found in @ref tests/graph/test_scheduler.cpp.
 *
 * References:
 *  * https://eel.is/c++draft/exec.sched
 */

namespace Tests::GraphImpl {

using graph_context_t = Kokkos::Execution::GraphContext<TEST_EXECUTION_SPACE>;
using graph_scheduler_t = Kokkos::Execution::GraphImpl::Scheduler<TEST_EXECUTION_SPACE>;
using graph_schedule_sender_t = typename graph_scheduler_t::Sender;
using graph_scheduler_attrs_t = typename graph_schedule_sender_t::Attributes;

//! @test @ref Kokkos::Execution::GraphImpl::Scheduler models the @c stdexec::scheduler concept.
static_assert(Tests::Utils::check_scheduler<graph_scheduler_t>());

/**
 * @test Check that querying the completion scheduler from a schedule sender of a scheduler returns the scheduler.
 *
 * See https://eel.is/c++draft/exec.sched#5.
 */
TEST(Scheduler, round_trip_property) {
    const graph_context_t ctx{TEST_EXECUTION_SPACE{}};
    const graph_scheduler_t sch = ctx.get_scheduler();
    ASSERT_EQ(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(stdexec::schedule(sch))), sch);
}

/**
 * @test Check that the @c schedule method of @ref Kokkos::Execution::GraphImpl::Scheduler returns
 *       @ref Kokkos::Execution::GraphImpl::Scheduler::Sender.
 */
consteval bool test_scheduler_schedule() {
    static_assert(
        std::same_as<decltype(stdexec::schedule(std::declval<const graph_scheduler_t&>())), graph_schedule_sender_t>);

    return true;
}
static_assert(test_scheduler_schedule());

/**
 * @test Check that the @c stdexec::get_env query on @ref Kokkos::Execution::GraphImpl::Scheduler::Sender returns
 *       @ref Kokkos::Execution::GraphImpl::Scheduler::Sender::Attributes.
 */
consteval bool test_schedule_sender_attrs() {
    static_assert(std::same_as<stdexec::env_of_t<graph_schedule_sender_t>, const graph_scheduler_attrs_t&>);

    return true;
}
static_assert(test_schedule_sender_attrs());

//! @test Check @ref Kokkos::Execution::GraphImpl::Scheduler queries.
consteval bool test_scheduler_queries() {
    static_assert(std::same_as<
                  stdexec::__query_result_t<graph_scheduler_t, stdexec::get_completion_domain_t<stdexec::set_value_t>>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    static_assert(
        std::same_as<
            stdexec::__query_result_t<graph_scheduler_t, stdexec::get_completion_scheduler_t<stdexec::set_value_t>>,
            graph_scheduler_t
        >);

    return true;
}
static_assert(test_scheduler_queries());

//! @test Check queries of @ref Kokkos::Execution::GraphImpl::Scheduler::Sender::Attributes.
consteval bool test_scheduler_attrs_queries() {
    static_assert(
        std::same_as<
            stdexec::__query_result_t<graph_scheduler_attrs_t, stdexec::get_completion_scheduler_t<stdexec::set_value_t>>,
            graph_scheduler_t
        >);

    static_assert(
        std::same_as<
            stdexec::__query_result_t<graph_scheduler_attrs_t, stdexec::get_completion_domain_t<stdexec::set_value_t>>,
            Kokkos::Execution::GraphImpl::Domain
        >);

    return true;
}
static_assert(test_scheduler_attrs_queries());

} // namespace Tests::GraphImpl
