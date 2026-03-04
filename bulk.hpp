#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/bulk.hpp"
#include "kokkos_ext/impl/execution_space/parallel_for.hpp"

namespace Kokkos::Experimental::details::execution_space {

template <>
struct transform_sender_for<stdexec::bulk_t> {
    template <
        typename Env,
        Kokkos::Experimental::details::impl::has_parallel_policy Data,
        execution_space_completing_sender<Env> Sndr
    >
    auto operator()(const Env& env, stdexec::bulk_t, Data&& data, Sndr&& sndr) const noexcept {
        auto [parallel_policy, shape, functor] = std::forward<Data>(data);

        using functor_t = decltype(functor);

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);
        auto exec = schd.state->exec;

        using execution_space = decltype(exec);

        Kokkos::RangePolicy<execution_space> policy(std::move(exec), 0, shape);
        std::string label(std::format("{}: bulk", Kokkos::Impl::TypeInfo<execution_space>::name()));

        return ParallelForSender<Sndr, functor_t, decltype(policy)>{
            {{std::move(label), std::move(functor), std::move(policy)}}, std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
