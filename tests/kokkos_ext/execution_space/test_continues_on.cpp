#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
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

using      execution_space = Kokkos::DefaultExecutionSpace;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class ContinuesOnTest : public impl::ExecutionSpaceContextTest<execution_space>,
                        public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check traits of the sender created by the customized @c continues_on.
TEST_F(ContinuesOnTest, traits)
{
    static_assert(::utils::check_continues_on<decltype(context_t{exec}.get_scheduler())>());
}

//! @test A @c then and a @c sync_wait following a @c continues_on properly use the execution space instance.
TEST_F(ContinuesOnTest, then_sync_wait)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    ::stdexec::sender auto chain = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
        )
    );

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when using it many times on the same execution space instance.
 *
 * There shouldn't be any fencing required in this case.
 */
TEST_F(ContinuesOnTest, transition_to_same_execution_space_instance)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::just()
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
        )
    );

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when transitioning from one execution space instance to another (of the same type).
 */
TEST_F(ContinuesOnTest, transition_to_another_execution_space_instance_and_back_same_type)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const context_t esc_A{exec_A}; SHOW_EXEC_SPACE_ID(exec_A)
    const context_t esc_B{exec_B}; SHOW_EXEC_SPACE_ID(exec_B)

    auto chain = ::stdexec::just()
        | ::stdexec::continues_on(esc_A.get_scheduler())
        | ADD_THEN
        | ::stdexec::continues_on(esc_B.get_scheduler())
        | ADD_THEN
        | ::stdexec::continues_on(esc_A.get_scheduler())
        | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    if(are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(recorded_events, ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec_A, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec_B, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec_A, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait"))
        ));
    } else {
        ASSERT_THAT(recorded_events, ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec_A, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR (exec_B, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR (exec_A, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait"))
        ));
    }

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when transitioning from one execution space instance to another (of different type).
 */
TEST_F(ContinuesOnTest, transition_to_another_execution_space_instance_and_back_different_type)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const host_execution_space exec_h{};

    const Kokkos::Experimental::ExecutionSpaceContext esc_h{exec_h};
    const context_t                                   esc  {exec};

    SHOW_EXEC_SPACE_ID(exec)
    SHOW_EXEC_SPACE_ID(exec_h)

    auto chain = ::stdexec::just()
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN
        | ::stdexec::continues_on(esc_h.get_scheduler())
        | ADD_THEN
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    if(are_same_instances(exec, exec_h)) {
        ASSERT_THAT(recorded_events, ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec,   dispatch_label(exec,   "then")),
            MATCHER_FOR_BEGIN_PFOR (exec_h, dispatch_label(exec_h, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec,   dispatch_label(exec,   "then")),
            MATCHER_FOR_BEGIN_FENCE(exec,   dispatch_label(exec,   "sync_wait"))
        ));
    } else {
        ASSERT_THAT(recorded_events, ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR (exec,   dispatch_label(exec,   "then")),
            MATCHER_FOR_BEGIN_FENCE(exec,   dispatch_label(exec,   "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR (exec_h, dispatch_label(exec_h, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec_h, dispatch_label(exec_h, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR (exec,   dispatch_label(exec,   "then")),
            MATCHER_FOR_BEGIN_FENCE(exec,   dispatch_label(exec,   "sync_wait"))
        ));
    }

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

//! @test No kernel launch happens, but @c stdexec::sync_wait fences when starting with @c stdexec::just_stopped.
TEST_F(ContinuesOnTest, just_stopped)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec}; SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::just_stopped()
        | ::stdexec::continues_on(esc.get_scheduler())
        | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    ASSERT_THAT(recorded_events, ::testing::ElementsAre(
        MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
    ));

    ASSERT_EQ(data(), 0) << "It should not execute on 'set_error'.";
}

} // namespace tests::kokkos_ext
