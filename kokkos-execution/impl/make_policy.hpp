#ifndef KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
#define KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP

#include "kokkos-execution/impl/iteration_space.hpp"

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec, extents Extents, typename... Properties>
requires(Extents::rank() == 1)
auto make_policy(const Extents& extents, const Properties&...) noexcept -> Kokkos::RangePolicy<Exec, Properties...> {
    return {0, extents.extent(0)};
}

template <Kokkos::ExecutionSpace Exec, extents Extents, typename... Properties>
requires(Extents::rank() == 0)
auto make_policy(const Exec& exec, const Extents&, const Properties&...) noexcept
    -> Kokkos::SinglePolicy<Exec, Properties...> {
    return {exec};
}

template <Kokkos::ExecutionSpace Exec, extents Extents, typename... Properties>
requires(Extents::rank() == 1)
auto make_policy(const Exec& exec, const Extents& extents, const Properties&...) noexcept
    -> Kokkos::RangePolicy<Exec, Properties...> {
    return {exec, 0, extents.extent(0)};
}

template <Kokkos::ExecutionSpace Exec, range_slice_with_default_stride RangeSlice>
auto make_policy(const Exec& exec, const RangeSlice& rgsl) noexcept -> Kokkos::RangePolicy<Exec> {
    return {exec, rgsl.first, rgsl.last};
}

template <Kokkos::ExecutionSpace Exec, typename ExecPolicy>
requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
auto make_policy(const Exec& exec, ExecPolicy&& policy) noexcept -> std::remove_cvref_t<ExecPolicy> {
    static_assert(
        std::same_as<Exec, typename std::remove_cvref_t<ExecPolicy>::execution_space>,
        "Only the execution space instance, not its type, can be bound late.");
    return {Kokkos::Impl::PolicyUpdate{}, std::forward<ExecPolicy>(policy), exec};
}

template <Kokkos::ExecutionSpace Exec, typename... Args>
using make_policy_t = decltype(make_policy(std::declval<const Exec&>(), std::declval<Args>()...));

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
