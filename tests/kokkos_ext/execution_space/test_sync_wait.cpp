#include "kokkos-utils/callbacks/Helpers.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/kokkos_ext/execution_space/Helpers.hpp"

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
    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler());

    Kokkos::utils::callbacks::Manager::initialize();

    ASSERT_THAT(
        Kokkos::utils::callbacks::RecorderListener<Kokkos::utils::callbacks::BeginFenceEvent>::record([chain = std::move(chain)] () mutable {
            const auto value = ::stdexec::sync_wait(std::move(chain));
            static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
            ASSERT_TRUE(value.has_value());
        }),
        ::testing::Contains(MATCHER_FOR_BEGIN_FENCE)
    );

    Kokkos::utils::callbacks::Manager::finalize();
}

/// This helper struct throws when @c Kokkos will copy construct it, see
/// https://github.com/kokkos/kokkos/blob/1e75c539491b8ce46c4671ce2e2275e15f1c27bc/core/src/Kokkos_Parallel.hpp#L142-L144
/// for instance. This ensures that we can catch in @c Kokkos::Experimental::details::execution_space::ThenReceiver::set_value.
struct Throws
{
    Throws() = default;
    Throws& operator=(const Throws&) = default;
    Throws(Throws&&) = default;
    Throws& operator=(Throws&&) = default;

    Throws(const Throws&) {
        throw std::runtime_error("throwing in copy constructor");
    }

    KOKKOS_FUNCTION
    void operator()() const {
        Kokkos::abort("This is not intended to be called.");
    }
};

/**
 * The matchers expect a @c const call operator, but the @ref chain
 * has to be moved into the @c sync_wait (so it has to be @c mutable).
 * This is not achievable with a lambda.
 */
template <typename T>
struct Mutable
{
    mutable T chain;

    void operator()() const {
        ::stdexec::sync_wait(std::move(chain));
    }
};

//! @test Check that @ref Kokkos::Experimental::details::execution_space::SyncWait properly rethrows if needed.
TEST_F(ExecutionSpaceContextTest, rethrows)
{
    const context_t esc{exec};

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::then(Throws{});

    ASSERT_THAT(
        Mutable{.chain = std::move(chain)},
        testing::ThrowsMessage<std::runtime_error>(testing::StrEq("throwing in copy constructor"))
    );
}

} // namespace tests::kokkos_ext
