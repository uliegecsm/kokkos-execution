#ifndef KOKKOS_EXECUTION_IMPL_QUERYABLE_HPP
#define KOKKOS_EXECUTION_IMPL_QUERYABLE_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

template <typename Query>
struct queryable_for {
    template <typename T, typename... Args>
    using type = stdexec::__mbool<stdexec::__queryable_with<T, Query, Args...>>;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_QUERYABLE_HPP
