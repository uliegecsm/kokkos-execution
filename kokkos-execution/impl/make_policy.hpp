#ifndef KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
#define KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

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
 * They are marked @c noexcept so that they can be used in @c noexcept expressions without themselves triggerting the expression to be potentially
 * throwing. The constructors of @c Kokkos execution policies abort if precondition checks fail, but are not marked @c noexcept.
 */
template <Kokkos::ExecutionSpace Exec, typename... Properties>
struct make_policy_t {
    [[nodiscard]]
    auto operator()(const Exec& exec) const noexcept -> Kokkos::SinglePolicy<Exec, Properties...> {
        return {exec};
    }

    template <extents Extents>
    requires(Extents::rank() == 1)
    [[nodiscard]]
    auto operator()(const Exec& exec, const Extents& extents) const noexcept
        -> Kokkos::RangePolicy<Exec, Properties...> {
        return {exec, 0, extents.extent(0)};
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

template <typename Exec, typename... Args>
struct MakePolicyResultType {
    using type = decltype(make_policy<Exec>(std::declval<const Exec&>(), std::declval<Args>()...));
};

template <typename Exec, typename... Properties, typename... Args>
struct MakePolicyResultType<Exec, properties<Properties...>, Args...> {
    using type = decltype(make_policy<Exec, Properties...>(std::declval<const Exec&>(), std::declval<Args>()...));
};

template <Kokkos::ExecutionSpace Exec, typename... Args>
using make_policy_result_t = typename MakePolicyResultType<Exec, Args...>::type;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
