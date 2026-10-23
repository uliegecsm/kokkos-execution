#ifndef KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
#define KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP

#include "kokkos-execution/impl/iteration_space.hpp"

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec, extents Extents>
requires(Extents::rank() == 1)
auto make_policy(const Exec& exec, const Extents& extents) -> Kokkos::RangePolicy<Exec> {
    return {exec, 0, extents.extent(0)};
}

template <Kokkos::ExecutionSpace Exec, range_slice_with_default_stride RangeSlice>
auto make_policy(const Exec& exec, const RangeSlice& rgsl) -> Kokkos::RangePolicy<Exec> {
    return {exec, rgsl.first, rgsl.last};
}

template <Kokkos::ExecutionSpace Exec, typename ExecPolicy>
requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
auto make_policy(const Exec& exec, ExecPolicy&& policy) -> std::remove_cvref_t<ExecPolicy> {
    static_assert(
        std::same_as<Exec, typename std::remove_cvref_t<ExecPolicy>::execution_space>,
        "Only the execution space instance, not its type, can be bound late.");
    return {Kokkos::Impl::PolicyUpdate{}, std::forward<ExecPolicy>(policy), exec};
}

template <Kokkos::ExecutionSpace Exec, typename... Args>
using make_policy_t = decltype(make_policy(std::declval<const Exec&>(), std::declval<Args>()...));

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
