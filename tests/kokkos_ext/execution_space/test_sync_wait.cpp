#include "tests/kokkos_ext/execution_space/Helpers.hpp"

#include "kokkos-utils/callbacks/Helpers.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c sync_wait by @c Kokkos::Experimental::ExecutionSpaceContext
 * -------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c sync_wait.
 *
 * The tests can be found in @ref kokkos_ext/execution_space/test_sync_wait.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using ExecutionSpaceContextTest = impl::ExecutionSpaceContextTest<execution_space>;

/**
 * @test Ensure that @c sync_wait is properly customized.
 *
 * Improperly customized @c sync_wait should result in a missing synchronization.
 *
 * @todo Use some @c Kokkos tools hook to ensure that proper fencing happens instead of relying on captured output.
 */
TEST_F(ExecutionSpaceContextTest, sync_wait)
{
    const context_t esc{*exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler());

    Kokkos::utils::callbacks::Manager::initialize();

    ASSERT_THAT(
        Kokkos::utils::callbacks::RecorderListener<Kokkos::utils::callbacks::BeginFenceEvent>::record([chain = std::move(chain)] () mutable {
            const auto value = ::stdexec::sync_wait(std::move(chain));
            static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
            ASSERT_TRUE(value.has_value());
        }),
        ::testing::Contains(
            ABeginFenceEvent(
                ::testing::Field(&Kokkos::utils::callbacks::BeginFenceEvent::name,   ::testing::StrEq(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<execution_space>::name()))),
                ::testing::Field(&Kokkos::utils::callbacks::BeginFenceEvent::dev_id, ::testing::Eq(Kokkos::Tools::Experimental::device_id(*exec)))
            )
        )
    );

    Kokkos::utils::callbacks::Manager::finalize();
}

} // namespace tests::kokkos_ext
