#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c then by @c Kokkos::Experimental::ExecutionSpaceContext
 * --------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c then.
 *
 * The tests can be found in @ref kokkos_ext/execution_space/test_then.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class ThenTest : public impl::ExecutionSpaceContextTest<execution_space>,
                 public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
};

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext does its duty well when used with @c then
 *       within a chain started with @c stdexec::schedule.
 */
TEST_F(ThenTest, then_schedule)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler()) | ADD_THEN | ADD_THEN;

    using chain_t = decltype(chain);

    //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler::Domain domain.
    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<chain_t>>,
        ::stdexec::default_domain
    >);
    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, chain_t>,
        Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<execution_space>::Domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
        decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(chain))),
        scheduler_t
    >);

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)] () mutable {
            ::stdexec::sync_wait(std::move(chain));
        }),
        ContainsInOrder<variant_t>(
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
        )
    );

    ASSERT_EQ(data(), 2);
}

/**
 * @test Similar to @ref tests::kokkos_ext::ThenTest_then_schedule_Test, but the chain is scheduled
 *       with a @c starts_on.
 */
TEST_F(ThenTest, then_starts_on)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    //! Create a chain that does not start with a schedule sender.
    auto chain = ::stdexec::just() | ADD_THEN | ADD_THEN;

    /// The chain cannot be queried for a completion scheduler.
    /// It may complete on the value channel or the error channel, since @c Kokkos may throw.
    /// It hasn't been connected yet, so the domain is indeterminate.
    using chain_t = decltype(chain);

    static_assert(!tests::stdexec::has_completion_scheduler_for<chain_t, ::stdexec::set_value_t>);
    static_assert(tests::stdexec::has_completion_signatures<chain_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);

    static_assert(std::same_as<
        ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, chain_t>,
        ::stdexec::indeterminate_domain<>
    >);

    //! Call @c starts_on.
    auto starts_on = ::stdexec::starts_on(esc.get_scheduler(), std::move(chain));

    using starts_on_t = decltype(starts_on);

    //! It has a completion scheduler for the value channel.
    static_assert(tests::stdexec::has_completion_scheduler_for<starts_on_t, ::stdexec::set_value_t>);
    static_assert(std::same_as<
        decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(starts_on))),
        scheduler_t
    >);

    //! Until it is connected, the completion signatures are *dependent* (they are not fully known yet).
    static_assert(std::same_as<std::invoke_result_t<
        ::stdexec::get_completion_signatures_t, starts_on_t>,
        ::stdexec::_ERROR_<::stdexec::dependent_sender_error, ::stdexec::__errs::_WITH_SENDER_<::stdexec::__basic_sender<::stdexec::__starts_on_ns::starts_on_t, Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<execution_space>, ::stdexec::__basic_sender<::stdexec::__then::then_t, tests::ThenFunctor<Kokkos::View<int, Kokkos::SharedSpace>>, ::stdexec::__basic_sender<::stdexec::__then::then_t, tests::ThenFunctor<Kokkos::View<int, Kokkos::SharedSpace>>, ::stdexec::__basic_sender<::stdexec::__just::just_t, ::stdexec::__tup::__tuple<>>>>>>>
    >);

    static_assert(std::same_as<std::invoke_result_t<
        ::stdexec::get_completion_signatures_t, starts_on_t, ::stdexec::env<>>,
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>
    >);

    /// We can compile and sync wait; the @c then are indeed launched using our customization of @c then.
    /// However, the result can't be verified yet because we haven't customized @c starts_on yet.
    /// We are therefore missing a fence, that we have to add manually.
    /// @todo Remove the manual fence once the @c starts_on is properly customized.
    ASSERT_THAT(
        recorder_listener_t::record([starts_on = std::move(starts_on)] () mutable {
            ::stdexec::sync_wait(std::move(starts_on));
        }),
        ContainsInOrder<variant_t>(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")))
    );

    exec.fence();
    ASSERT_EQ(data(), 2);
}

/**
 * @test Check the sequence of events when a @c Kokkos::View is captured in a chain of work.
 *
 * This test ensures that the deallocation happens after the kernels are finished.
 */
TEST_F(ThenTest, then_lifetime)
{
    //! Create the chain in a scope.
    auto create_chain_in_scope = [&](){
        const view_s_t data(Kokkos::view_alloc("data - shared space", exec));

        const context_t esc{exec};

        return ::stdexec::schedule(esc.get_scheduler()) | ADD_THEN | ADD_THEN;
    };

    //! Run the whole test in a lambda.
    auto run_test = [&](){
        auto chain = create_chain_in_scope();

        using chain_t = decltype(chain);

        //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler::Domain domain.
        static_assert(std::same_as<
            ::stdexec::__domain_of_t<::stdexec::env_of_t<chain_t>>,
            ::stdexec::default_domain
        >);
        static_assert(std::same_as<
            ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, chain_t>,
            Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<execution_space>::Domain
        >);

        //! It has a completion scheduler for the value channel.
        static_assert(std::same_as<
            decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(chain))),
            scheduler_t
        >);

        ::stdexec::sync_wait(std::move(chain));
    };

    ASSERT_THAT(
        recorder_listener_t::record(run_test),
        ContainsInOrder<variant_t>(
            Kokkos::utils::callbacks::AAllocateDataEvent(
                ::testing::Field(
                    &Kokkos::utils::callbacks::AllocateDataEvent::alloc,
                    ::testing::Field(
                        &Kokkos::utils::callbacks::AllocDescriptor::name,
                        ::testing::StrEq("data - shared space")
                    )
                )
            ),
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")),
            Kokkos::utils::callbacks::ADeallocateDataEvent(
                ::testing::Field(
                    &Kokkos::utils::callbacks::DeallocateDataEvent::alloc,
                    ::testing::Field(
                        &Kokkos::utils::callbacks::AllocDescriptor::name,
                        ::testing::StrEq("data - shared space")
                    )
                )
            )
        )
    );
}

} // namespace tests::kokkos_ext
