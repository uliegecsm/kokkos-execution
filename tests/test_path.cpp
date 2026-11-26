#include "gtest/gtest.h"

#include "kokkos-execution/parallel_for.hpp"
#include "kokkos-execution/path.hpp"

#include "tests/utils/category.hpp"
#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"

/**
 * @addtogroup unittests
 *
 * Sender adaptor @c Kokkos::Execution::path
 * -----------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::path.
 *
 * The tests can be found in @ref tests/test_path.cpp.
 */

namespace Tests {

//! @test The @ref Kokkos::Execution::path_t operator may be used in a constant expression.
consteval bool test_path_in_constant_expression() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    [[maybe_unused]]
    const auto sndr_a = Kokkos::Execution::path(Kokkos::Execution::parallel_for(42, functor_t{}));

    [[maybe_unused]]
    const auto sndr_b = Kokkos::Execution::path(
        Kokkos::Execution::parallel_for(Kokkos::range_slice<size_t, size_t>{42, 43}, functor_t{}));

    return std::is_constant_evaluated();
}

static_assert(test_path_in_constant_expression());

template <typename T>
struct TEST_CATEGORY(PathCopiesAndMovesTest)
    : public virtual testing::Test {
    using counter_t = Tests::Utils::Functors::Counter;

    static constexpr typename T::value_type N = T::value; // NOLINT(readability-identifier-length)

    template <std::size_t... Idx>
    requires(sizeof...(Idx) > 0)
    stdexec::sender auto make_sndr_impl(std::index_sequence<Idx...>) {
        return stdexec::just() | Kokkos::Execution::path(Kokkos::Execution::parallel_for(Idx, counter)...);
    }

    stdexec::sender auto make_sndr() {
        return make_sndr_impl(std::make_index_sequence<N>{});
    }

    counter_t counter{};
};

TYPED_TEST_SUITE_P(TEST_CATEGORY(PathCopiesAndMovesTest));

//! @test Check how many copies and moves are needed to build a sender of @ref Kokkos::Execution::path_t.
TYPED_TEST_P(TEST_CATEGORY(PathCopiesAndMovesTest), build_sender) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const stdexec::sender auto sndr = this->make_sndr();

    CHECK_COUNTER_COPIES_AND_MOVES(EXPECT, 0, this->N, 0, 2 * this->N)
}

REGISTER_TYPED_TEST_SUITE_P(TEST_CATEGORY(PathCopiesAndMovesTest), build_sender);

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

} // namespace Tests
