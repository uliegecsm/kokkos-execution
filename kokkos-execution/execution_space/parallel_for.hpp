#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/operation_state.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/parallel_for.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForClosure {
    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    Kokkos::Execution::Impl::ParallelForData<Functor, policy_t> data;

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

    parallel_for_t tag;
    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelForSender, stdexec::set_error_t(std::exception_ptr))

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<OpState<Sndr, Rcvr, closure_t>, Sndr&&, Rcvr&&, closure_t&&>)
        -> OpState<Sndr, Rcvr, closure_t> {
        return OpState<Sndr, Rcvr, closure_t>(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <>
struct TransformSenderFor<Kokkos::Execution::parallel_for_t> {
    template <typename Env, typename Data, execution_space_completing_sender<Env> Sndr>
    auto operator()(const Env& env, Kokkos::Execution::parallel_for_t, Data&& data, Sndr&& sndr) const noexcept {
        auto [label, functor, policy] = std::forward<Data>(data);

        using functor_t = decltype(functor);
        using policy_t = decltype(policy);
        using closure_t = ParallelForClosure<functor_t, policy_t>;

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        //! Only the execution space instance, not its type, can be bound lately.
        static_assert(
            std::same_as<typename decltype(schd)::execution_space, typename closure_t::execution_space>,
            "The policy's execution space type must be the same as the completion scheduler's execution space type.");

        return ParallelForSender<Sndr, functor_t, policy_t>{
            {},
            {{std::move(label),
              std::move(functor),
              policy_t(Kokkos::Impl::PolicyUpdate{}, std::move(policy), schd.state->exec)}},
            std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#if !STDEXEC_HAS_BUILTIN(__builtin_structured_binding_size)
namespace stdexec {

//! See also https://cor3ntin.github.io/posts/clang21/#__builtin_structured_binding_size.
template <stdexec::sender Sndr, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
inline constexpr auto __structured_binding_size_v< // NOLINT(bugprone-reserved-identifier)
    Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<Sndr, Functor, ExecPolicy>
> = 3;

} // namespace stdexec
#endif

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP
