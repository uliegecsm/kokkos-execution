#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/operation_state.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/execution_space/sender_introspection.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/parallel_for.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForClosure {
    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    Kokkos::Execution::Impl::ParallelForData<Label, Functor, policy_t> data;

    void execute() const & {
        if constexpr (std::convertible_to<Label, std::string>) {
            Kokkos::parallel_for(data.label, data.policy, data.functor);
        } else {
            Kokkos::parallel_for(std::string{data.label}, data.policy, data.functor);
        }
    }

    const policy_t& get_policy() const & noexcept {
        return data.policy;
    }
};

template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelForClosure<Label, Functor, ExecPolicy>;
    using execution_space = typename closure_t::execution_space;

    parallel_for_t tag;
    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelForSender, stdexec::set_error_t(std::exception_ptr))

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(noexcept(
        MakeOpStateFn<Sndr, Rcvr, closure_t>{}(std::declval<Sndr>(), std::declval<Rcvr>(), std::declval<closure_t>())))
        -> opstate_t<Sndr, Rcvr, closure_t> {
        return MakeOpStateFn<Sndr, Rcvr, closure_t>{}(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <>
struct TransformSenderFor<Kokkos::Execution::parallel_for_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelForSender<
        Sndr,
        typename std::remove_cvref_t<Data>::label_t,
        typename std::remove_cvref_t<Data>::functor_t,
        typename std::remove_cvref_t<Data>::policy_t
    >;

    template <typename Env, typename Data, execution_space_completing_sender<Env> Sndr>
    auto operator()(const Env& env, Kokkos::Execution::parallel_for_t, Data&& data, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 parallel_for_t,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        auto [label, functor, policy] = std::forward<Data>(data);

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        //! Only the execution space instance, not its type, can be bound lately.
        static_assert(
            std::same_as<
                typename decltype(schd)::execution_space,
                typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t::execution_space
            >,
            "The policy's execution space type must be the same as the completion scheduler's execution space type.");

        return trnsfrmd_sndr_t<Env, Data, Sndr>{
            parallel_for_t{},
            {{std::move(label),
              std::move(functor),
              typename std::remove_cvref_t<Data>::policy_t(
                  Kokkos::Impl::PolicyUpdate{}, std::move(policy), schd.state->exec)}},
            std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#if !STDEXEC_HAS_BUILTIN(__builtin_structured_binding_size)
namespace stdexec {

//! See also https://cor3ntin.github.io/posts/clang21/#__builtin_structured_binding_size.
template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
inline constexpr auto __structured_binding_size_v< // NOLINT(bugprone-reserved-identifier)
    Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<Sndr, Label, Functor, ExecPolicy>
> = 3;

} // namespace stdexec
#endif

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP
