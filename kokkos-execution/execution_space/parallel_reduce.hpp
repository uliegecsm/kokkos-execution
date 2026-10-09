#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/operation_state.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/parallel_reduce.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy, typename Reducer>
struct ParallelReduceClosure {
    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    Kokkos::Execution::Impl::ParallelReduceData<Label, Functor, policy_t, Reducer> data;

    void submit() const & {
        if constexpr (std::convertible_to<Label, std::string>) {
            Kokkos::parallel_reduce(data.label, data.policy, data.functor, data.reducer);
        } else {
            Kokkos::parallel_reduce(std::string{data.label}, data.policy, data.functor, data.reducer);
        }
    }

    const policy_t& get_policy() const & noexcept {
        return data.policy;
    }
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
        typename std::remove_cvref_t<Data>::policy_t,
        typename std::remove_cvref_t<Data>::reducer_t
    >;

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env& env, Kokkos::Execution::parallel_reduce_t, Data&& data, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        if constexpr (execution_space_completing_sender<Sndr, Env>) {
            auto [label, functor, policy, reducer] = std::forward<Data>(data);

            const auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

            //! Only the execution space instance, not its type, can be bound lately.
            static_assert(
                std::same_as<
                    typename decltype(schd)::execution_space,
                    typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t::execution_space
                >,
                "The policy's execution space type must be the same as the completion scheduler's execution space "
                "type.");

            return trnsfrmd_sndr_t<Env, Data, Sndr>{
                {{std::move(label),
                  std::move(functor),
                  typename std::remove_cvref_t<Data>::policy_t(
                      Kokkos::Impl::PolicyUpdate{}, std::move(policy), schd.state->exec),
                  std::move(reducer)}},
                std::forward<Sndr>(sndr)};
        } else {
            return no_execution_space_scheduler_in_env<parallel_reduce_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
