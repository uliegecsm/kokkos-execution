#include "gtest/gtest.h"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/category.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/counter.hpp"
#include "tests/utils/sink_receiver.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c Kokkos::Execution::path by @c Kokkos::Execution::ExecutionSpaceContext
 * ------------------------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::ExecutionSpaceImpl::PathSender.
 *
 * The tests can be found in @ref tests/execution_space/test_path.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

template <typename T>
struct TEST_CATEGORY(PathCopiesAndMovesTest)
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE> {
    using counter_t = Tests::Utils::Functors::Counter;

    static constexpr typename T::value_type N = T::value; // NOLINT(readability-identifier-length)

    template <std::size_t... Idx>
    requires(sizeof...(Idx) > 0)
    stdexec::sender auto make_sndr_impl(std::index_sequence<Idx...>) {
        return stdexec::schedule(esc.get_scheduler())
             | Kokkos::Execution::path(Kokkos::Execution::parallel_for(Idx, counter)...);
    }

    stdexec::sender auto make_sndr() {
        return make_sndr_impl(std::make_index_sequence<N>{});
    }

    Kokkos::Execution::ExecutionSpaceContext<TEST_EXECUTION_SPACE> esc{this->exec};
    counter_t counter{};
};

TYPED_TEST_SUITE_P(TEST_CATEGORY(PathCopiesAndMovesTest));

//! @test TODO
TYPED_TEST_P(TEST_CATEGORY(PathCopiesAndMovesTest), transform_sender) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr = stdexec::transform_sender(this->make_sndr(), stdexec::env<>{});

    CHECK_COUNTER_COPIES_AND_MOVES(EXPECT, 0, this->N, 0, 5 * this->N)
}

//! @test TODO
TYPED_TEST_P(TEST_CATEGORY(PathCopiesAndMovesTest), connect) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const stdexec::operation_state auto opstate = stdexec::connect(this->make_sndr(), Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_COPIES_AND_MOVES(EXPECT, 0, this->N, 0, 6 * this->N)
}

REGISTER_TYPED_TEST_SUITE_P(TEST_CATEGORY(PathCopiesAndMovesTest), transform_sender, connect);

using path_copies_and_moves_types_t = ::testing::Types<
    std::integral_constant<size_t, 1>,
    std::integral_constant<size_t, 2>,
    std::integral_constant<size_t, 3>,
    std::integral_constant<size_t, 4>,
    std::integral_constant<size_t, 5>,
    std::integral_constant<size_t, 6>,
    std::integral_constant<size_t, 9>
>;

STDEXEC_PP_EVAL(
    INSTANTIATE_TYPED_TEST_SUITE_P,
    TEST_CATEGORY(PathCopiesAndMovesTestInstantiation),
    TEST_CATEGORY(PathCopiesAndMovesTest),
    path_copies_and_moves_types_t);

} // namespace Tests::ExecutionSpaceImpl
