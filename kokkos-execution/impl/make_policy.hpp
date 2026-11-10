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

template <typename T>
struct UnderlyingIndex {
    using type = T;
};

template <typename T>
requires Kokkos::detail::is_constant_wrapper<T>
struct UnderlyingIndex<T> {
    using type = typename T::value_type;
};

//! The value type behind a constant wrapper, or the type itself.
template <typename T>
using underlying_index_t = typename UnderlyingIndex<T>::type;

template <typename>
struct IndexTypeFor;

template <extents T>
struct IndexTypeFor<T> {
    using type = typename T::index_type;
};

template <typename FirstType, typename LastType, typename StrideType>
struct IndexTypeFor<Kokkos::range_slice<FirstType, LastType, StrideType>> {
    static_assert(
        std::same_as<underlying_index_t<FirstType>, underlying_index_t<LastType>>,
        "Kokkos::range_slice first and last must have the same index type.");

    using type = underlying_index_t<LastType>;
};

//! Get the index type for @p IterSpace.
template <typename IterSpace>
using index_type_for_t = typename IndexTypeFor<IterSpace>::type;

template <typename IterSpace, typename Tag, typename Exec, template <typename...> typename PolicyType, typename... Args>
struct AddIndexType {
    /**
     * According to https://eel.is/c++draft/exec#bulk-4, the callable is invoked using an index
     * whose type is the same as the @c stdexec::bulk_t shape argument type.
     *
     * See also https://github.com/NVIDIA/stdexec/pull/2359.
     */
    using type = std::conditional_t<
        std::same_as<Tag, stdexec::bulk_t>,
        PolicyType<Exec, Kokkos::IndexType<index_type_for_t<IterSpace>>, Args...>,
        PolicyType<Exec, Args...>
    >;
};

//! If required, add the index type to the @c PolicyType.
template <typename IterSpace, typename Tag, typename Exec, template <typename...> typename PolicyType, typename... Args>
using add_index_type_t = typename AddIndexType<IterSpace, Tag, Exec, PolicyType, Args...>::type;

/**
 * @brief Utility functions that wrap constructors of @c Kokkos execution policies.
 *
 * They are marked @c noexcept so that they can be used in @c noexcept expressions without themselves triggering the expression to be potentially
 * throwing. The constructors of @c Kokkos execution policies abort if precondition checks fail, but are not marked @c noexcept.
 */
template <typename Tag, Kokkos::ExecutionSpace Exec, typename... Properties>
struct make_policy_t {
    [[nodiscard]]
    auto operator()(const Exec& exec) const noexcept
        -> Kokkos::SinglePolicy<Exec, Properties...> requires std::same_as<Tag, stdexec::then_t>
    {
        return {exec};
    }

    template <extents Extents>
    requires(Extents::rank() == 1)
    [[nodiscard]]
    auto operator()(const Exec& exec, const Extents& extents) const noexcept
        -> add_index_type_t<Extents, Tag, Exec, Kokkos::RangePolicy, Properties...> {
        return {exec, 0, extents.extent(0)};
    }

    template <static_range_slice_with_default_stride RangeSlice>
    requires(
        std::remove_cvref_t<decltype(RangeSlice::last)>::value - std::remove_cvref_t<decltype(RangeSlice::first)>::value
        == 1)
    [[nodiscard]]
    auto operator()(const Exec& exec, const RangeSlice& rgsl) const noexcept
        -> add_index_type_t<RangeSlice, Tag, Exec, Kokkos::RangePolicy, Kokkos::LaunchBounds<1>, Properties...> {
        return {exec, rgsl.first, rgsl.last};
    }

    template <range_slice_with_default_stride RangeSlice>
    [[nodiscard]]
    auto operator()(const Exec& exec, const RangeSlice& rgsl) const noexcept
        -> add_index_type_t<RangeSlice, Tag, Exec, Kokkos::RangePolicy, Properties...> {
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

template <typename Tag, Kokkos::ExecutionSpace Exec, typename... Properties>
inline constexpr make_policy_t<Tag, Exec, Properties...> make_policy{};

template <typename MakePolicy, typename... Args>
requires stdexec::__is_instance_of<std::remove_cvref_t<MakePolicy>, make_policy_t>
using make_policy_result_t = std::invoke_result_t<const MakePolicy&, Args...>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_MAKE_POLICY_HPP
