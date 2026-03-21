#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/labeled.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c bulk by @c Kokkos::Execution::ExecutionSpaceContext
 * -----------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c bulk.
 *
 * The tests can be found in @ref tests/execution_space/test_bulk.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class BulkTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check traits of sender returned by @c bulk when customized for @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename BulkTest::schedule_sender_t;

    //! Bulk sender.
    using functor_t = Tests::Utils::Functors::SumIndices<BulkTest::view_s_t>;
    using bulk_sndr_t = stdexec::transform_sender_result_t<
        decltype(stdexec::bulk(std::declval<schd_sndr_t>(), stdexec::par, 1, std::declval<functor_t>())),
        stdexec::env<>
    >;

    static_assert(std::same_as<
                  bulk_sndr_t,
                  Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<
                      schd_sndr_t,
                      functor_t,
                      Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
                  >
    >);

    return true;
}
static_assert(test_sndr_traits());

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_no_throw_transformable() {
    using sndr_bulk_t =
        decltype(stdexec::schedule(std::declval<typename BulkTest::scheduler_t>()) | stdexec::bulk(stdexec::par, 1, Tests::Utils::Functors::NoOp<false, false, false>{}));

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::set_value_t,
                  sndr_bulk_t&&,
                  stdexec::env<>
    >);

    using sndr_bulk_maythrow_on_move_t =
        decltype(stdexec::schedule(std::declval<typename BulkTest::scheduler_t>()) | stdexec::bulk(stdexec::par, 1, Tests::Utils::Functors::NoOp<false, false, true>{}));

    //! @todo It should not be nothrow transformable.
    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::set_value_t,
                  sndr_bulk_maythrow_on_move_t&&,
                  stdexec::env<>
    >);

    return true;
}
static_assert(test_sndr_no_throw_transformable());

/**
 * @test Check construction of operation state from a parallel for sender
 *       when passing the sender as a @c const reference.
 */
constexpr bool test_op_state_passed_by_const_ref() {
    using sndr_t =
        decltype(stdexec::schedule(std::declval<typename BulkTest::context_t>().get_scheduler()) | stdexec::bulk(stdexec::par, 1, Tests::Utils::Functors::Labeled<'a'>{}));

    static_assert(!std::is_const_v<sndr_t>);

    //! Connect the sender as a @c const reference.
    using op_state_from_sndr_const_ref_t = stdexec::connect_result_t<const sndr_t&, Tests::Utils::SinkReceiver>;

    static_assert(std::same_as<
                  op_state_from_sndr_const_ref_t,
                  Kokkos::Execution::ExecutionSpaceImpl::OpState<
                      const typename BulkTest::schedule_sender_t&,
                      Tests::Utils::SinkReceiver,
                      Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
                          Tests::Utils::Functors::Labeled<'a'>,
                          Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
                      >
                  >
    >);

    return true;
}
static_assert(test_op_state_passed_by_const_ref());

//! @test Check that @ref Kokkos::Execution::ExecutionSpaceContext does its duty well when used with @c bulk.
TEST_F(BulkTest, bulk) {
    constexpr size_t size = 10;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler()) | BULK_SUM_INDICES(size, data);

    using chain_t = decltype(chain);

    //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain domain.
    static_assert(std::same_as<stdexec::__domain_of_t<stdexec::env_of_t<chain_t>>, stdexec::default_domain>);
    static_assert(std::same_as<
                  stdexec::__detail::__completing_domain_t<stdexec::set_value_t, chain_t>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
                  decltype(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(chain))),
                  scheduler_t
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), size / 2 * (size - 1));
}

} // namespace Tests::ExecutionSpaceImpl
