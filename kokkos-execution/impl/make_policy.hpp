#ifndef KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
#define KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#    include "Kokkos_Core.hpp"
#endif

#include "kokkos-execution/impl/iteration_space.hpp"
#include "kokkos-execution/impl/properties.hpp"

namespace Kokkos::Execution::Impl {

//! Only the execution space instance, not its type, can be bound late.
template <typename ExecPolicy, typename Exec>
concept updatable_to = Kokkos::ExecutionPolicy<ExecPolicy> && std::same_as<Exec, typename ExecPolicy::execution_space>
                    && requires(const ExecPolicy& policy, const Exec& exec) {
                           ExecPolicy{Kokkos::Impl::PolicyUpdate{}, policy, exec};
                       };

/**
 * @brief Utility functions that wrap constructors of @c Kokkos execution policies.
 *
 * They are marked @c noexcept so that they can be used in @c noexcept expressions without themselves triggering the expression to be potentially
 * throwing. The constructors of @c Kokkos execution policies abort if precondition checks fail, but are not marked @c noexcept.
 */
template <Kokkos::ExecutionSpace Exec, typename... Properties>
struct make_policy_t {
    [[nodiscard]]
    auto operator()(const Exec& exec) const noexcept -> Kokkos::SinglePolicy<Exec, Properties...> {
        return {exec};
    }

    template <extents Extents>
    requires(Extents::rank() == 1 && SenderTag == bulk_t)
    [[nodiscard]]
    auto operator()(const Exec& exec, const Extents& extents) const noexcept
        -> Kokkos::RangePolicy<Exec, IndexType<...>, Properties...> {
        return {exec, 0, extents.extent(0)};
    }

    template <extents Extents>
    requires(Extents::rank() == 1)
    [[nodiscard]]
    auto operator()(const Exec& exec, const Extents& extents) const noexcept
        -> Kokkos::RangePolicy<Exec, Properties...> {
        return {exec, 0, extents.extent(0)};
    }

    template <static_range_slice_with_default_stride RangeSlice>
    requires(
        std::remove_cvref_t<decltype(RangeSlice::first)>::value == 0
        && std::remove_cvref_t<decltype(RangeSlice::last)>::value == 1)
    [[nodiscard]]
    auto operator()(const Exec& exec, const RangeSlice&) const noexcept {
        return this->operator()(exec);
    }

    template <static_range_slice_with_default_stride RangeSlice>
    requires(
        std::remove_cvref_t<decltype(RangeSlice::first)>::value != 0
        && std::remove_cvref_t<decltype(RangeSlice::last)>::value
                   - std::remove_cvref_t<decltype(RangeSlice::first)>::value
               == 1)
    [[nodiscard]]
    auto operator()(const Exec& exec, const RangeSlice& rgsl) const noexcept
        -> Kokkos::RangePolicy<Exec, Kokkos::LaunchBounds<1>, Properties...> {
        return {exec, rgsl.first, rgsl.last};
    }

    template <range_slice_with_default_stride RangeSlice>
    [[nodiscard]]
    auto operator()(const Exec& exec, const RangeSlice& rgsl) const noexcept
        -> Kokkos::RangePolicy<Exec, Properties...> {
        return {exec, rgsl.first, rgsl.last};
    }

    template <updatable_to<Exec> ExecPolicy>
    requires(sizeof...(Properties) == 0)
    [[nodiscard]]
    auto operator()(const Exec& exec, const ExecPolicy& policy) const noexcept -> ExecPolicy {
        return ExecPolicy{Kokkos::Impl::PolicyUpdate{}, policy, exec};
    }

    static Exec default_instance() noexcept {
        return Exec{};
    }

    template <typename... Args>
    requires(!Kokkos::ExecutionSpace<std::remove_cvref_t<Args>> && ...)
    [[nodiscard]]
    auto operator()(Args&&... args) const
        STDEXEC_AUTO_RETURN(this->operator()(default_instance(), std::forward<Args>(args)...))
};

template <Kokkos::ExecutionSpace Exec, typename... Properties>
inline constexpr make_policy_t<Exec, Properties...> make_policy{};

template <typename MakePolicy, typename... Args>
requires stdexec::__is_instance_of<std::remove_cvref_t<MakePolicy>, make_policy_t>
using make_policy_result_t = std::invoke_result_t<const MakePolicy&, Args...>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
