#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/operation_state.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/make_policy.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/parallel_reduce.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy, typename Reducer>
struct ParallelReduceClosure {
    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    void submit() const & {
        if constexpr (std::convertible_to<Label, std::string>) {
            Kokkos::parallel_reduce(label, policy, functor, reducer);
        } else {
            Kokkos::parallel_reduce(std::string{label}, policy, functor, reducer);
        }
    }

    const policy_t& get_policy() const & noexcept {
        return policy;
    }

    Label label;
    Functor functor;
    policy_t policy;
    Reducer reducer;
};

template <
    typename Tag,
    stdexec::sender Sndr,
    typename Label,
    typename Functor,
    Kokkos::ExecutionPolicy ExecPolicy,
    typename Reducer
>
struct ParallelReduceSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelReduceClosure<Label, Functor, ExecPolicy, Reducer>;
    using execution_space = typename closure_t::execution_space;

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelReduceSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(noexcept(make_op_state_t<Sndr, Rcvr, closure_t>{}(
        std::declval<Sndr>(),
        std::declval<Rcvr>(),
        std::declval<closure_t>()))) -> op_state_t<Sndr, Rcvr, closure_t> {
        return make_op_state_t<Sndr, Rcvr, closure_t>{}(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <>
struct TransformSenderFor<Kokkos::Execution::parallel_reduce_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelReduceSender<
        parallel_reduce_t,
        Sndr,
        typename std::remove_cvref_t<Data>::label_t,
        typename std::remove_cvref_t<Data>::functor_t,
        Impl::make_policy_t<Impl::exec_of_t<Sndr, Env>, typename std::remove_cvref_t<Data>::policy_or_iter_space_t>,
        typename std::remove_cvref_t<Data>::reducer_t
    >;

    template <typename Env, typename Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        Kokkos::Execution::parallel_reduce_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        const auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return trnsfrmd_sndr_t<Env, Data, Sndr>{
            {.label = stdexec::__forward_like<Data>(data.label),
             .functor = stdexec::__forward_like<Data>(data.functor),
             .policy = Impl::make_policy(schd.state->exec, stdexec::__forward_like<Data>(data.policy_or_iter_space)),
             .reducer = stdexec::__forward_like<Data>(data.reducer)},
            std::forward<Sndr>(sndr)
        };
    }

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, Kokkos::Execution::parallel_reduce_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<parallel_reduce_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
