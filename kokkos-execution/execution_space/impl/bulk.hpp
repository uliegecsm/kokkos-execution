#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_BULK_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_BULK_HPP

#include "kokkos-execution/execution_space/Context_fwd.hpp"
#include "kokkos-execution/execution_space/impl/parallel_for.hpp"
#include "kokkos-execution/impl/bulk.hpp"

namespace Kokkos::Execution::execution_space::impl {

template <>
struct transform_sender_for<stdexec::bulk_t> {
    template <
        typename Env,
        Kokkos::Execution::impl::has_parallel_policy Data,
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

} // namespace Kokkos::Execution::execution_space::impl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_BULK_HPP
