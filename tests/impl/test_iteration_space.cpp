#include "gtest/gtest.h"

#include "kokkos-execution/impl/iteration_space.hpp"

/**
 * @addtogroup unittests
 *
 * Iteration space
 * ---------------
 *
 * This group of tests check the utilities provided in @ref kokkos-execution/impl/iteration_space.hpp.
 *
 * The tests can be found in @ref tests/impl/test_iteration_space.cpp.
 */

namespace Tests {

using namespace Kokkos::Execution::Impl;

//! @test Check that @ref Kokkos::Execution::Impl::range_slice_with_default_stride works as expected.
consteval bool test_range_slice_with_default_stride() {
    //! A runtime stride does not model the concept.
    static_assert(!range_slice_with_default_stride<Kokkos::range_slice<size_t, size_t, int>>);

    //! A defaulted stride models the concept.
    static_assert(range_slice_with_default_stride<Kokkos::range_slice<size_t, size_t>>);

    //! A compile-time stride that is not equal to one does not model the concept.
    static_assert(!range_slice_with_default_stride<Kokkos::range_slice<int, int, std::integral_constant<int, 0>>>);
    static_assert(range_slice_with_default_stride<Kokkos::range_slice<int, int, std::integral_constant<int, 1>>>);
    static_assert(!range_slice_with_default_stride<Kokkos::range_slice<int, int, std::integral_constant<int, 2>>>);

    return true;
}

static_assert(test_range_slice_with_default_stride());

//! @test Check that @ref Kokkos::Execution::Impl::static_range_slice_with_default_stride works as expected.
consteval bool test_static_range_slice_with_default_stride() {
    using value_t = std::integral_constant<size_t, 0>;

    //! A runtime stride does not model the concept.
    static_assert(static_range_slice_with_default_stride<Kokkos::range_slice<value_t, value_t>>);
    static_assert(!static_range_slice_with_default_stride<Kokkos::range_slice<value_t, value_t, int>>);

    //! A defaulted stride models the concept.
    static_assert(!static_range_slice_with_default_stride<Kokkos::range_slice<int, int>>);
    static_assert(!static_range_slice_with_default_stride<Kokkos::range_slice<int, value_t>>);
    static_assert(!static_range_slice_with_default_stride<Kokkos::range_slice<value_t, int>>);
    static_assert(static_range_slice_with_default_stride<Kokkos::range_slice<value_t, value_t>>);

    //! A compile-time stride that is not equal to one does not model the concept.
    static_assert(
        !static_range_slice_with_default_stride<Kokkos::range_slice<value_t, value_t, std::integral_constant<int, 0>>>);
    static_assert(
        static_range_slice_with_default_stride<Kokkos::range_slice<value_t, value_t, std::integral_constant<int, 1>>>);
    static_assert(
        !static_range_slice_with_default_stride<Kokkos::range_slice<value_t, value_t, std::integral_constant<int, 2>>>);

    return true;
}

static_assert(test_static_range_slice_with_default_stride());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests
