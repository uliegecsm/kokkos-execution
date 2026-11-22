#include <array>

#include "kokkos-execution/execution_space.hpp"
#include "kokkos-execution/parallel_for.hpp"

#include "tests/utils/functors/sum_indices.hpp"

/**
 * @addtogroup unittests
 *
 * Moves implied by the composition of senders are optimized away
 * ---------------------------------------------------------------
 *
 * Composing a chain of @c N sender adaptors performs @c N(N+3)/2 moves of functors.
 *
 * For a functor whose members are @c Kokkos::View, a move has no side effect. It leaves the moved-from view
 * untracked, hence the destructor of a moved-from view does not decrement any reference count.
 * In an optimized build, all these moves and destructors are thus expected to boil down to
 * initializing the result from the functors that were passed in.
 *
 * This cannot be observed at run time, so this file is only compiled, for a given @c CHAIN_LENGTH.
 * The test then looks at the object file. In the function below, every @c Kokkos::View destructor is the one
 * of a moved-from view. Every functor comes from the caller, and every functor that is alive at the end
 * is returned to the caller. If the object file still references the function
 * that decrements the reference count of an allocation, then the compiler lost track of some moved-from object.
 *
 * The tests can be found in @ref tests/codegen/elide_moves.cpp.
 */

#if !defined(CHAIN_LENGTH)
#    error "CHAIN_LENGTH must be defined."
#endif

namespace Tests::Codegen {

using execution_space = Kokkos::DefaultExecutionSpace;

using context_t = Kokkos::Execution::ExecutionSpaceContext<execution_space>;
using scheduler_t = decltype(std::declval<const context_t>().get_scheduler());

using value_t = int;
using view_s_t = Kokkos::View<value_t, Kokkos::SharedSpace>;
using functor_t = Tests::Utils::Functors::SumIndices<view_s_t>;

using array_of_functors_t = std::array<functor_t, CHAIN_LENGTH>;

auto compose(const scheduler_t schd, array_of_functors_t& array_of_functors) {
    return std::apply(
        [&](auto&... functors) {
            return (stdexec::schedule(schd) | ... | Kokkos::Execution::parallel_for(10, std::move(functors)));
        },
        array_of_functors);
}

} // namespace Tests::Codegen
