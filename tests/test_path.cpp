#include "kokkos-execution/parallel_for.hpp"
#include "kokkos-execution/path.hpp"

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
    const auto sndr_a = Kokkos::Execution::path({Kokkos::Execution::parallel_for_t, 42, functor_t{}});

    [[maybe_unused]]
    const auto sndr_b = Kokkos::Execution::path({
        Kokkos::Execution::parallel_for_t, Kokkos::range_slice<size_t, size_t>{42, 43},
          functor_t{}
    });

    return std::is_constant_evaluated();
}

static_assert(test_path_in_constant_expression());

} // namespace Tests
