#ifndef KOKKOS_EXECUTION_IMPL_BULK_HPP
#define KOKKOS_EXECUTION_IMPL_BULK_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

//! See https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__bulk.hpp#L100.
template <typename Data>
concept has_parallel_policy = requires(const Data& data) {
    { data.__pol_ } -> std::same_as<const stdexec::__bulk::__policy_wrapper<stdexec::parallel_policy>&>;
};

//! Extract the policy, shape and functor type of @c bulk data.
template <typename...>
struct BulkTraits;

template <typename Policy, typename Shape, typename Functor>
struct BulkTraits<stdexec::__bulk::__data<Policy, Shape, Functor>> {
    using policy_t = Policy;
    using shape_t = Shape;
    using functor_t = Functor;
};

template <typename Data>
using bulk_traits = BulkTraits<std::remove_cvref_t<Data>>;

/**
 * According to https://eel.is/c++draft/exec#bulk-4, the callable is invoked using an index
 * whose type is the same as the @c stdexec::bulk_t shape argument type.
 *
 * See also https://github.com/NVIDIA/stdexec/pull/2359.
 */
template <typename Env, typename Data, typename Sndr>
using policy_for_bulk_t =
    Kokkos::RangePolicy<Impl::exec_of_t<Sndr, Env>, Kokkos::IndexType<typename bulk_traits<Data>::shape_t>>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_BULK_HPP
