#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/execution_space/sender_introspection.hpp"
#include "kokkos-execution/impl/bulk.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct TransformSenderFor<stdexec::bulk_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelForSender<
        Sndr,
        Impl::ParallelForData<
            typename Kokkos::Execution::Impl::bulk_traits<Data>::functor_t,
            Kokkos::RangePolicy<exec_of_t<Sndr, Env>>,
            Impl::Label
        >
    >;

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 parallel_for_t,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        auto& [parallel_policy, shape, functor] = data;

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return trnsfrmd_sndr_t<Env, Data, Sndr>{
            parallel_for_t{},
            {{{std::string(Impl::dispatch_label<exec_of_t<Sndr, Env>, ": bulk">())},
              stdexec::__forward_like<Data>(functor),
              Kokkos::RangePolicy<exec_of_t<Sndr, Env>>(schd.state->exec, 0, shape)}},
            std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
