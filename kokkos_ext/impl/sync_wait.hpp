#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNC_WAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNC_WAIT_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"

namespace Kokkos::Experimental::details::impl {
struct env {
    ::stdexec::run_loop::scheduler schd;

    [[nodiscard]]
    auto query(::stdexec::get_scheduler_t) const noexcept -> ::stdexec::run_loop::scheduler {
        return schd;
    }

    [[nodiscard]]
    auto query(::stdexec::get_delegation_scheduler_t) const noexcept -> ::stdexec::run_loop::scheduler {
        return schd;
    }
};

} // namespace Kokkos::Experimental::details::impl

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNC_WAIT_HPP
