#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_BULK_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_BULK_HPP

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::impl {

//! See https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__bulk.hpp#L100.
template <typename Data>
concept parallel_policy = requires(const Data& data) {
    { data.__pol_ } -> std::same_as<const ::stdexec::__bulk::__policy_wrapper<::stdexec::parallel_policy>&>;
};

} // namespace Kokkos::Experimental::details::impl

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_BULK_HPP
