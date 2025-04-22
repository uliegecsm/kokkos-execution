#include "kokkos-utils/callbacks/RecorderListener.hpp"

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
                 public Kokkos::utils::callbacks::ManagerTestFixture
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent>;
};

/**
 * @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext does its duty well when used with @c then.
 *       In this case, the scheduler is already known when @c then is called, such that our customization of
 *       @c ExecutionSpaceScheduler::Domain::transform_sender for @c then is called readily (early).
 */
TEST_F(ThenTest, then_early_customization)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler()) | ADD_THEN | ADD_THEN;

    using chain_t = decltype(chain);

    //! The chain's environment cannot be queried for its domain.
    static_assert(!::stdexec::tag_invocable<::stdexec::get_domain_t, ::stdexec::env_of_t<chain_t>>);

    //! However, it has a completion scheduler for the value channel.
    static_assert(::stdexec::__has_completion_scheduler<chain_t, ::stdexec::set_value_t>);

    static_assert(std::same_as<
        ::stdexec::__detail::__completion_scheduler_for<::stdexec::env_of_t<chain_t>, ::stdexec::set_value_t>,
        scheduler_t
    >);

    //! Therefore, it has a **non-default early** completion domain.
    static_assert(std::same_as<
        ::stdexec::__detail::__completion_domain_of<chain_t>,
        scheduler_domain_t
    >);

    static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_t>, scheduler_domain_t>);

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
 * @test Similar to @ref tests::kokkos_ext::ThenTest_then_early_customization_Test, but the completion scheduler is not known
 *       until @c starts_on is called (late).
 */
TEST_F(ThenTest, then_late_customization)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    //! Create a chain that does not start with a schedule sender.
    auto chain = ::stdexec::just() | ADD_THEN | ADD_THEN;

    /// The chain's last sender cannot be queried for a completion scheduler. See also
    /// https://github.com/NVIDIA/stdexec/blob/b888185d667f68b9a8bda5d0c81d03edf9ec3fe1/include/stdexec/__detail/__env.hpp#L212-L215.
    /// It may complete on the value channel or the error channel, since @c Kokkos may throw.
    /// It hasn't been customized yet, so the early domain is the default one.
    using chain_t = decltype(chain);

    static_assert(!::stdexec::__has_completion_scheduler<chain_t, ::stdexec::set_value_t>);
    static_assert(tests::stdexec::has_completion_signatures<chain_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);

    static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_t>, ::stdexec::default_domain>);

    //! Call @c starts_on.
    auto starts_on = ::stdexec::starts_on(esc.get_scheduler(), std::move(chain));

    using starts_on_t = decltype(starts_on);

    /// We are still not able to query the completion scheduler, and the completion signatures are still both the value and error channels.
    static_assert(!::stdexec::__has_completion_scheduler<starts_on_t, ::stdexec::set_value_t>);
    static_assert(tests::stdexec::has_completion_signatures<starts_on_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);

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

        //! The chain's environment cannot be queried for its domain.
        static_assert(!::stdexec::tag_invocable<::stdexec::get_domain_t, ::stdexec::env_of_t<chain_t>>);

        //! However, it has a completion scheduler for the value channel.
        static_assert(::stdexec::__has_completion_scheduler<chain_t, ::stdexec::set_value_t>);

        static_assert(std::same_as<
            ::stdexec::__detail::__completion_scheduler_for<::stdexec::env_of_t<chain_t>, ::stdexec::set_value_t>,
            scheduler_t
        >);

        //! Therefore, it has a **non-default early** completion domain.
        static_assert(std::same_as<
            ::stdexec::__detail::__completion_domain_of<chain_t>,
            scheduler_domain_t
        >);

        static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_t>, scheduler_domain_t>);

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
