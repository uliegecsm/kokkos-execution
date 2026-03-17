#ifndef KOKKOS_EXECUTION_IMPL_ATTRIBUTES_HPP
#define KOKKOS_EXECUTION_IMPL_ATTRIBUTES_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Filter queries on a sender to allow forwarding queries only.
 *
 * See also:
 *  * https://github.com/NVIDIA/stdexec/blob/5076be2b35de2e78330201b888d82c81b8cb428b/include/nvexec/stream/common.cuh#L455
 *  * https://github.com/NVIDIA/stdexec/blob/5076be2b35de2e78330201b888d82c81b8cb428b/include/stdexec/__detail/__on.hpp#L97
 */
template <stdexec::sender Sndr>
struct ForwardingAttributes {
    template <stdexec::__forwarding_query Query, typename... Args>
    requires stdexec::__queryable_with<stdexec::env_of_t<Sndr>, Query, Args...>
    [[nodiscard]]
    constexpr auto query(Query, Args&&... args) const
        noexcept(stdexec::__nothrow_queryable_with<stdexec::env_of_t<Sndr>, Query, Args...>)
            -> stdexec::__query_result_t<stdexec::env_of_t<Sndr>, Query, Args...> {
        return stdexec::__query<Query>()(stdexec::get_env(sndr), std::forward<Args>(args)...);
    }

    Sndr const & sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(_type_, _obj_)                                             \
    [[nodiscard]]                                                                                                      \
    constexpr auto get_env() const noexcept -> Kokkos::Execution::Impl::ForwardingAttributes<_type_> {                 \
        return {_obj_};                                                                                                \
    }

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_ATTRIBUTES_HPP
