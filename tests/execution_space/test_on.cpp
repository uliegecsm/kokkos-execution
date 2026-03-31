#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_scheduler_type.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/kokkos.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c on by @c Kokkos::Execution::ExecutionSpaceContext
 * ---------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c on.
 *
 * The tests can be found in @ref tests/execution_space/test_on.cpp.
 */

using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class OnTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<EventDiscardMatcher<TEST_EXECUTION_SPACE>, BeginFenceEvent, BeginParallelForEvent>;
};

/**
 * @test Check that @ref Kokkos::Execution::ExecutionSpaceContext supports @c on, using the same execution space instance.
 *
 * There shouldn't be any fencing required in this case.
 */
TEST_F(OnTest, on_same_execution_space_instance) {
    const view_s_t data(Kokkos::view_alloc("data", exec));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler())
               | Tests::Utils::check_scheduler_type<stdexec::set_value_t, scheduler_t>() | THEN_INCREMENT(data)
               | stdexec::on(
                     esc.get_scheduler(),
                     Tests::Utils::check_scheduler_type<stdexec::set_value_t, scheduler_t>() | THEN_INCREMENT(data))
               | Tests::Utils::check_scheduler_type<stdexec::set_value_t, scheduler_t>() | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test Check that @ref Kokkos::Execution::ExecutionSpaceContext supports @c on, using different execution space instances
 *       of the same type.
 *
 * Proper fencing is required when transitioning from one execution space instance to another.
 */
TEST_F(OnTest, on_another_execution_space_instance_same_type) {
    const view_s_t data(Kokkos::view_alloc("data", exec));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const context_t esc_A{exec_A};
    Tests::Utils::show_exec_space_id(exec_A, "exec_A");
    const context_t esc_B{exec_B};
    Tests::Utils::show_exec_space_id(exec_B, "exec_B");

    auto chain = stdexec::schedule(esc_A.get_scheduler()) | THEN_INCREMENT(data)
               | stdexec::on(esc_B.get_scheduler(), THEN_INCREMENT(data)) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); });

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait"))));
    } else {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait"))));
    }

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test Check that @ref Kokkos::Execution::ExecutionSpaceContext supports @c on, using different execution space instances
 *       of different types.
 *
 * Proper fencing is required when transitioning from one execution space instance to another.
 */
TEST_F(OnTest, many_execution_space_instances_of_different_type) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const host_execution_space exec_h{};

    const Kokkos::Execution::ExecutionSpaceContext esc_h{exec_h};
    const context_t esc{exec};

    Tests::Utils::show_exec_space_id(exec, "exec");
    Tests::Utils::show_exec_space_id(exec_h, "exec_h");

    auto chain = stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data)
               | stdexec::on(esc_h.get_scheduler(), THEN_INCREMENT(data)) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); });

    if (Tests::Utils::are_same_instances(exec, exec_h)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    } else {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_h, dispatch_label(exec_h, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    }

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

} // namespace Tests::ExecutionSpaceImpl
