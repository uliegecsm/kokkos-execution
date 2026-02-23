#include <bit>

#include "kokkos_ext/impl/execution_space/parallel_for.hpp"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c Kokkos::Experimental::parallel_for by @c Kokkos::Experimental::ExecutionSpaceContext
 * --------------------------------------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Experimental::details::execution_space::ParallelForSender.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_parallel_for.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class ParallelForTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

/**
 * @test Check traits of sender returned by @c Kokkos::Experimental::parallel_for either uncustomized
 *       or customized for @c Kokkos::Experimental::details::execution_space::Domain.
 */
template <template <typename, typename, typename> class SndrAdptr>
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename ParallelForTest::schedule_sender_t;

    //! Parallel for sender.
    using functor_t = BulkFunctor<typename ParallelForTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<execution_space>;
    using pfor_sndr_t = SndrAdptr<schd_sndr_t, functor_t, policy_t>;

    //! Models the sender concept.
    static_assert(::stdexec::sender<pfor_sndr_t>);

    //! Has the expected completion signatures.
    using completion_signatures_t = ::stdexec::__completion_signatures_of_t<pfor_sndr_t, ::stdexec::env<>>;

    static_assert(::stdexec::__mset_eq<
                  ::stdexec::__mset<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>,
                  completion_signatures_t
    >);

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, pfor_sndr_t, ::stdexec::env<>>,
                  Kokkos::Experimental::details::execution_space::Domain
    >);

    //! Has the expected completion scheduler.
    static_assert(std::same_as<
                  ::stdexec::__completion_scheduler_of_t<::stdexec::set_value_t, pfor_sndr_t, ::stdexec::env<>>,
                  Kokkos::Experimental::details::execution_space::Scheduler<execution_space>
    >);

    //! Is connectable.
    static_assert(::stdexec::sender_to<pfor_sndr_t, tests::stdexec::SinkReceiver>);

    static_assert(std::same_as<
                  ::stdexec::transform_sender_result_t<pfor_sndr_t, ::stdexec::env_of_t<tests::stdexec::SinkReceiver>>,
                  Kokkos::Experimental::details::execution_space::ParallelForSender<schd_sndr_t, functor_t, policy_t>
    >);

    /**
     * It is not no throw connectable because the @c ParallelForClosure is not no throw move constructible.
     * This is so because it holds a functor that holds a @c Kokkos::View, and the latter are not no throw
     * move constructible.
     *
     * See also https://github.com/kokkos/kokkos/pull/8792.
     */
    static_assert(!std::is_nothrow_move_constructible_v<typename ParallelForTest::view_s_t>);
    static_assert(!std::is_nothrow_move_constructible_v<functor_t>);
    using closure_t = Kokkos::Experimental::details::execution_space::ParallelForClosure<functor_t, policy_t>;
    static_assert(!std::is_nothrow_move_constructible_v<closure_t>);
    static_assert(!::stdexec::__nothrow_connectable<pfor_sndr_t, tests::stdexec::SinkReceiver>);

    return true;
}
static_assert(test_sndr_traits<Kokkos::Experimental::ParallelForSender>());
static_assert(test_sndr_traits<Kokkos::Experimental::details::execution_space::ParallelForSender>());

//! @test Check decomposition of @ref Kokkos::Experimental::ParallelForSender into the algorithm tag, data, and child sender.
consteval bool test_sndr_decomposition() {
    //! Schedule sender.
    using schd_sndr_t = typename ParallelForTest::schedule_sender_t;

    //! Parallel for sender.
    using functor_t = BulkFunctor<typename ParallelForTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<execution_space>;
    using pfor_sndr_t = Kokkos::Experimental::ParallelForSender<schd_sndr_t, functor_t, policy_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(std::same_as<::stdexec::tag_of_t<pfor_sndr_t>, Kokkos::Experimental::parallel_for_t>);

    static_assert(
        std::same_as<::stdexec::__data_of<pfor_sndr_t>, Kokkos::Experimental::ParallelForData<functor_t, policy_t>>);

    static_assert(::stdexec::__nbr_children_of<pfor_sndr_t> == 1);
    static_assert(std::same_as<::stdexec::__child_of<pfor_sndr_t>, schd_sndr_t>);

    //! Is transformable via @c Kokkos::Experimental::details::execution_space::transform_sender_for.
    static_assert(
        ::stdexec::__applicable<
            Kokkos::Experimental::details::execution_space::transform_sender_for<::stdexec::tag_of_t<pfor_sndr_t>>,
            pfor_sndr_t,
            const ::stdexec::env<>&
        >);

    return true;
}
static_assert(test_sndr_decomposition());

//! @test Check traits of @ref Kokkos::Experimental::details::execution_space::ParallelForClosure.
template <typename ViewType, bool ExpectNoThrowMoveConstructible>
consteval bool test_closure_traits() {
    using functor_t = BulkFunctor<ViewType>;
    using policy_t = Kokkos::RangePolicy<execution_space>;
    using closure_t = Kokkos::Experimental::details::execution_space::ParallelForClosure<functor_t, policy_t>;

    //! Models the @ref Kokkos::Experimental::details::execution_space::Closure concept.
    static_assert(Kokkos::Experimental::details::execution_space::Closure<closure_t>);

    static_assert(std::is_nothrow_move_constructible_v<closure_t> == ExpectNoThrowMoveConstructible);

    return true;
}
static_assert(test_closure_traits<typename ParallelForTest::view_s_t, false>());
static_assert(test_closure_traits<std::span<int>, true>());

//! @test Check @ref Kokkos::Experimental::parallel_for with a team policy.
TEST_F(ParallelForTest, team_policy) {
    constexpr int size = 32;

    const auto [num_teams, team_size] = [&]() {
//! @c Kokkos hardcodes a maximum team size of 1 on @c HPX. See also https://github.com/kokkos/kokkos/blob/1c6efc105c2366a95fa3d0012b38bbf4c03aecd5/core/src/HPX/Kokkos_HPX.hpp#L742-L747.
#if defined(KOKKOS_ENABLE_HPX)
        if constexpr (std::same_as<execution_space, Kokkos::Experimental::HPX>) {
            return std::make_tuple(size, 1);
        }
#endif
        const int team_size_ = std::bit_floor(static_cast<unsigned short>(std::min(exec.concurrency(), size / 2)));
        return std::make_tuple(size / team_size_, team_size_);
    }();

    ASSERT_EQ(team_size * num_teams, size);

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
               | Kokkos::Experimental::parallel_for(
                     "hello from pfor",
                     Kokkos::TeamPolicy<execution_space>(num_teams, team_size),
                     BulkFunctor{.data = witness});

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness(), size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Experimental::parallel_for closure object creation overloads.
TEST_F(ParallelForTest, closure_object_creation_overloads) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
               | Kokkos::Experimental::parallel_for(
                     "passing label, execution policy and functor",
                     Kokkos::RangePolicy<execution_space>(0, size),
                     BulkFunctor{.data = witness})
               | Kokkos::Experimental::parallel_for(
                     Kokkos::RangePolicy<execution_space>(0, size), BulkFunctor{.data = witness})
               | Kokkos::Experimental::parallel_for(
                     "passing label, work count and functor", size, BulkFunctor{.data = witness})
               | Kokkos::Experimental::parallel_for(size, BulkFunctor{.data = witness});

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, "passing label, execution policy and functor"),
            MATCHER_FOR_BEGIN_PFOR(exec, Kokkos::Impl::TypeInfo<BulkFunctor<view_s_t>>::name()),
            MATCHER_FOR_BEGIN_PFOR(exec, "passing label, work count and functor"),
            MATCHER_FOR_BEGIN_PFOR(exec, Kokkos::Impl::TypeInfo<BulkFunctor<view_s_t>>::name()),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(witness(), 4 * size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Experimental::parallel_for with two consecutive parallel regions and check there is no fence in between.
TEST_F(ParallelForTest, two_parallel_regions) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
               | Kokkos::Experimental::parallel_for(
                     std::format("{}: hello from pfor", Kokkos::Impl::TypeInfo<execution_space>::name()),
                     Kokkos::RangePolicy<execution_space>(0, size),
                     BulkFunctor{.data = witness})
               | ::stdexec::then(
                     tests::utils::LoadCheckAddFunctor<value_t, on_device>{
                         .prev = size / 2 * (size - 1), .value = 4, .data = witness.data()})
               | Kokkos::Experimental::parallel_for(
                     std::format("{}: hello again from pfor", Kokkos::Impl::TypeInfo<execution_space>::name()),
                     Kokkos::RangePolicy<execution_space>(0, 2 * size),
                     BulkFunctor{.data = witness});

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "hello from pfor")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "hello again from pfor")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(witness(), size / 2 * (size - 1) + 4 + 2 * size * (2 * size - 1) / 2);
}

} // namespace tests::kokkos_ext
