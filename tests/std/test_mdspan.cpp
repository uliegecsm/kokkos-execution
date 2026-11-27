#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#if __has_include(<mdspan>)
#    include <mdspan>
#endif
#include <numeric>

#if !defined(__cpp_lib_mdspan)
#    include "Kokkos_Core.hpp"
#endif

/**
 * @c std::mdspan
 * --------------
 *
 * This group of tests check the behavior of @c std::mdspan.
 *
 * The tests can be found in @ref tests/std/test_mdspan.cpp.
 */

#if defined(__cpp_lib_mdspan)
using std::extents;
using std::mdspan;
#else
using Kokkos::extents;
using Kokkos::mdspan;
#endif

namespace Test {

//! @test @c std::mdspan can be used in a constant expression.
consteval bool test_mdspan() {
    std::array<double, 6> data{1, 2, 3, 4, 5, 6};

    const mdspan<const double, extents<int, 6>> data_as_mdspan{data.data()};

    double sum = 0.;

    for (std::size_t index = 0; index < data.size(); ++index)
        sum += data_as_mdspan[index]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    return sum == 21.;
}

static_assert(test_mdspan());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Test
