#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/diagnostics.hpp"
#include "kokkos-execution/execution_space/operation_state.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/make_policy.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/parallel_reduce.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <Kokkos::ExecutionSpace Exec, Impl::parallel_reduce_data Data>
struct ParallelReduceClosure {
    using execution_space = Exec;

    void submit() const & {
        if constexpr (Impl::labeled<Data>) {
            Kokkos::parallel_reduce(
                Impl::as_convertible_to_string(Impl::get_property<Impl::Label>(data)),
                this->get_policy(),
                data.functor,
                data.reducer);
        } else {
            Kokkos::parallel_reduce(this->get_policy(), data.functor, data.reducer);
        }
    }

    [[nodiscard]]
    auto get_policy() const & noexcept {
        return Impl::make_policy<execution_space>(this->get_exec(), data.policy_or_iter_space);
    }

    [[nodiscard]]
    auto get_exec() const & noexcept -> const execution_space& {
        return state->exec;
    }

    const Impl::State<execution_space>* state;
    Data data;
};

template <typename Tag, stdexec::sender Sndr, Kokkos::ExecutionSpace Exec, Impl::parallel_reduce_data Data>
struct ParallelReduceSender {
    using sender_concept = stdexec::sender_tag;

    using execution_space = Exec;
    using closure_t = ParallelReduceClosure<execution_space, Data>;

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
    using trnsfrmd_sndr_t =
        ParallelReduceSender<parallel_reduce_t, Sndr, Impl::exec_of_t<Sndr, Env>, std::remove_cvref_t<Data>>;

    template <typename Env, typename Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        Kokkos::Execution::parallel_reduce_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        STDEXEC_AUTO_RETURN(
            trnsfrmd_sndr_t<Env, Data, Sndr>{
                {.state = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env).state,
                 .data = std::forward<Data>(data)},
                std::forward<Sndr>(sndr)
    })

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, Kokkos::Execution::parallel_reduce_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<parallel_reduce_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
