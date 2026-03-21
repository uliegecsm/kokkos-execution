#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/impl/bulk.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct TransformSenderFor<stdexec::bulk_t> {
    template <typename Sndr, typename Env>
    using schd_t = stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, const Env&>;

    template <typename Sndr, typename Env>
    using execution_space = typename schd_t<Sndr, Env>::execution_space;

    template <typename Sndr, typename Env>
    using policy_t = Kokkos::RangePolicy<execution_space<Sndr, Env>>;

    template <typename Data>
    using functor_t = typename Kokkos::Execution::Impl::bulk_traits<Data>::functor_t;

    template <typename Sndr, typename Data, typename Env>
    using sndr_t = ParallelForSender<Sndr, functor_t<Data>, policy_t<Sndr, Env>>;

    template <
        typename Env,
        Kokkos::Execution::Impl::has_parallel_policy Data,
        execution_space_completing_sender<Env> Sndr
    >
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 sndr_t<Sndr, Data, Env>,
                 parallel_for_t,
                 typename sndr_t<Sndr, Data, Env>::closure_t&&,
                 Sndr&&
        >) {
        auto& [parallel_policy, shape, functor] = data;

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return sndr_t<Sndr, Data, Env>{
            {},
            {{std::string(std::format("{}: bulk", Kokkos::Impl::TypeInfo<execution_space<Sndr, Env>>::name())),
              stdexec::__forward_like<Data>(functor),
              policy_t<Sndr, Env>(schd.state->exec, 0, shape)}},
            std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
