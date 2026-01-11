#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Both @c stdexec::split and @c stdexec::when_all are supported by @c Kokkos::Experimental::ExecutionSpaceContext
 * ---------------------------------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly works with both
 * @c stdexec::split and @c stdexec::when_all.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_split.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class SplitTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t = std::variant<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Use @c stdexec::split and @c stdexec::sync_wait right after.
TEST_F(SplitTest, split_and_sync_wait) {
    const context_t esc{exec};

    ::stdexec::sender auto chain = ::stdexec::schedule(esc.get_scheduler()) | ::stdexec::split();

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::IsEmpty());
}

/**
 * @test Each branch of a @c stdexec::when_all that executes on @ref Kokkos::Experimental::ExecutionSpaceContext must synchronize before calling
 *       @c stdexec::set_value of the "end of the branch receiver".
 */
TEST_F(SplitTest, within) {
    const view_sa_t data(Kokkos::view_alloc(exec, "data - shared space"));

    ::exec::static_thread_pool pool{4};
    const context_t esc{exec};

    ::stdexec::sender auto fork = ::stdexec::schedule(pool.get_scheduler()) | ::stdexec::split();

    auto branch_a = fork | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN | ADD_THEN;
    auto branch_b = fork | ::stdexec::continues_on(pool.get_scheduler()) | ADD_THEN;
    auto branch_c = std::move(fork) | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN | ADD_THEN;

    auto chain = ::stdexec::when_all(std::move(branch_a), std::move(branch_b), std::move(branch_c))
               | ::stdexec::then([&data]() {
                     if (data() != 5)
                         Kokkos::abort("Synchronization issue.");
                 });

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    /// Each branch may be executed by a distinct host thread. However, the callback manager is not thread safe.
    /// So ordering of the event is not guaranteed.
    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::UnorderedElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation"))));

    ASSERT_EQ(data(), 5);
}

} // namespace tests::kokkos_ext
