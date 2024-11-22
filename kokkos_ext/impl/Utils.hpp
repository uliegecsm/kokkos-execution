#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_UTILS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_UTILS_HPP

#include "Kokkos_Core.hpp"

namespace Kokkos::Experimental::graph::details
{
/**
 * @brief Create a new @c Kokkos::RangePolicy from @p policy with an updated execution space instance.
 *
 * @note We cannot update the @p policy object for now, we have to create a new one.
 */
template <typename Policy, typename Exec>
requires (Kokkos::Impl::is_specialization_of<std::remove_cvref_t<Policy>, Kokkos::RangePolicy>::value && Kokkos::is_execution_space_v<std::remove_cvref_t<Exec>>)
Policy update_policy(const Policy& policy, Exec&& exec) {
    return Policy(std::forward<Exec>(exec), policy.begin(), policy.end(), Kokkos::ChunkSize(policy.chunk_size()));
}

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_UTILS_HPP
