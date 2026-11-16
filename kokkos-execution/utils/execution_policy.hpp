#ifndef KOKKOS_EXECUTION_UTILS_EXECUTION_POLICY_HPP
#define KOKKOS_EXECUTION_UTILS_EXECUTION_POLICY_HPP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

/**
 * @file
 *
 * This file contains utilities to work with @c Kokkos execution policies.
 */

namespace Kokkos::Execution::Utils {

//! Extract the @c Kokkos execution policy properties and execution space types.
template <typename>
struct ExtractExecutionPolicyProperties;

template <Kokkos::ExecutionSpace Exec, typename... Properties>
struct ExtractExecutionPolicyProperties<Kokkos::RangePolicy<Exec, Properties...>> {
    using execution_space = Exec;
    using properties_t = Kokkos::Impl::type_list<Properties...>;
};

//! Rebind a @c Kokkos execution policy properties.
template <typename...>
struct RebindExecutionPolicy;

template <
    template <typename...> typename ExecPolicyType,
    Kokkos::ExecutionSpace Exec,
    typename... OldProperties,
    typename... NewProperties
>
struct RebindExecutionPolicy<ExecPolicyType<Exec, OldProperties...>, Exec, Kokkos::Impl::type_list<NewProperties...>> {
    using type = ExecPolicyType<Exec, NewProperties...>;
};

//! @c true if @p T is an @c Kokkos::IndexType.
template <typename T>
struct is_index_type {
    static constexpr bool value = stdexec::__is_instance_of<T, Kokkos::IndexType>;
};

} // namespace Kokkos::Execution::Utils

#endif // KOKKOS_EXECUTION_UTILS_EXECUTION_POLICY_HPP
