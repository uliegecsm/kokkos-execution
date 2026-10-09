#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_REDUCE_INDICES_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_REDUCE_INDICES_HPP

#include <concepts>
#include <type_traits>

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Tests::Utils::Functors {

//! Reduce indices. @note To be used with @c Kokkos::parallel_reduce.
struct ReduceIndices {
    template <std::integral T, typename U>
    KOKKOS_FUNCTION void operator()(const T index, U& value) const {
        value += index;
    }
};

/**
 * @brief Similar to @ref ReduceIndices, with a team policy.
 *
 * @todo Because of the deduction logic in https://github.com/kokkos/kokkos/blob/dd9eb811539af4913ae8d4f29e17ab0c73676550/core/src/impl/Kokkos_FunctorAnalysis.hpp#L258-L303,
 *       we cannot template the team-handle call operator.
 *       Therefore, the member and value types must be known.
 *       See also https://github.com/kokkos/kokkos/issues/7794.
 *       Once solve, this functor can be fused with @ref ReduceIndices.
 */
template <Kokkos::ExecutionSpace Exec, typename ValueType>
struct ReduceIndicesWithTeam {
    using team_handle_t = typename Kokkos::TeamPolicy<Exec>::member_type;

    KOKKOS_FUNCTION void operator()(const team_handle_t& team_handle, ValueType& value) const {
        const auto start_index = team_handle.league_rank() * team_handle.team_size();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_handle, team_handle.team_size()),
            [&]<std::integral T>(const T index) { value += start_index + index; });
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_REDUCE_INDICES_HPP
