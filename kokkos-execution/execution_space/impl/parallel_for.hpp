#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_PARALLEL_FOR_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/execution_space/impl/context_fwd.hpp"
#include "kokkos-execution/execution_space/impl/operation_state.hpp"
#include "kokkos-execution/parallel_for.hpp"

namespace Kokkos::Execution::execution_space::impl {

template <typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForClosure {
    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    Kokkos::Execution::impl::ParallelForData<Functor, policy_t> data;

    void execute() const & {
        Kokkos::parallel_for(data.label, data.policy, data.functor);
    }

    const policy_t& get_policy() const & noexcept {
        return data.policy;
    }
};

template <stdexec::sender Sndr, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForSender {
    using sender_concept = stdexec::sender_t;

    using closure_t = ParallelForClosure<Functor, ExecPolicy>;
    using execution_space = typename closure_t::execution_space;

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelForSender, stdexec::set_error_t(std::exception_ptr))

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<OpState<Sndr, Rcvr, closure_t>, Sndr&&, Rcvr&&, closure_t&&>)
        -> OpState<Sndr, Rcvr, closure_t> {
        return OpState<Sndr, Rcvr, closure_t>(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Sndr, sndr)
};

template <>
struct transform_sender_for<Kokkos::Execution::parallel_for_t> {
    template <typename Env, typename Data, execution_space_completing_sender<Env> Sndr>
    auto operator()(const Env& env, Kokkos::Execution::parallel_for_t, Data&& data, Sndr&& sndr) const noexcept {
        auto [label, functor, policy] = std::forward<Data>(data);

        using functor_t = decltype(functor);
        using policy_t = decltype(policy);

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        policy_t policy_updated(Kokkos::Impl::PolicyUpdate{}, std::move(policy), schd.state->exec);
        return ParallelForSender<Sndr, functor_t, policy_t>{
            {{std::move(label), std::move(functor), std::move(policy_updated)}}, std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::execution_space::impl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_PARALLEL_FOR_HPP
