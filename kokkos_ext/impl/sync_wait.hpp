#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNC_WAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNC_WAIT_HPP

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::impl {
//! Inspired by https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__sync_wait.hpp#L45-L65.
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

//! Inspired by https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__sync_wait.hpp#L83-L86.
struct State {
    std::exception_ptr error;
    stdexec::run_loop loop;
};

} // namespace Kokkos::Experimental::details::impl

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNC_WAIT_HPP
