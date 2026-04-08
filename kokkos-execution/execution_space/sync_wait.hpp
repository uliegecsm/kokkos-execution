#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SYNC_WAIT_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/sync_wait.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

/**
 * @brief Customize @c sync_wait.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/e8a6a7b25fbc2463e1dfe0ee20973b1fe622bfcf/include/nvexec/stream_context.cuh#L247-L251
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-dispatch
 */
template <>
struct ApplySenderFor<stdexec::sync_wait_t> {
    template <execution_space_completing_sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept(std::is_nothrow_invocable_v<Impl::SyncWait::SyncWait, Sndr&&>) {
        return Impl::SyncWait::SyncWait{}(std::forward<Sndr>(sndr));
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SYNC_WAIT_HPP
