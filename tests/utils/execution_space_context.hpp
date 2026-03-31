#ifndef KOKKOS_EXECUTION_TESTS_UTILS_EXECUTION_SPACE_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_EXECUTION_SPACE_CONTEXT_HPP

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/context.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"

namespace Tests::Utils {

template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpaceContextTest : public ContextTest<Kokkos::Execution::ExecutionSpaceContext, Exec> { };

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_EXECUTION_SPACE_CONTEXT_HPP
