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

template <Impl::parallel_reduce_data Data>
struct ParallelReduceClosure {
    using policy_t = typename Data::policy_or_iter_space_t;
    static_assert(Kokkos::ExecutionPolicy<policy_t>);
    using execution_space = typename policy_t::execution_space;

    void submit() const & {
        if constexpr (Impl::labeled<Data>) {
            Kokkos::parallel_reduce(
                Impl::as_convertible_to_string(Impl::get_property<Impl::Label>(data)),
                data.policy_or_iter_space,
                data.functor,
                data.reducer);
        } else {
            Kokkos::parallel_reduce(data.policy_or_iter_space, data.functor, data.reducer);
        }
    }

    [[nodiscard]]
    const policy_t& get_policy() const & noexcept {
        return data.policy_or_iter_space;
    }

    Data data;
};

template <typename Tag, stdexec::sender Sndr, Impl::parallel_reduce_data Data>
struct ParallelReduceSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelReduceClosure<Data>;
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
    using data_t = Impl::parallel_reduce_data_from_t<
        typename std::remove_cvref_t<Data>::functor_t,
        Impl::make_policy_t<Impl::exec_of_t<Sndr, Env>, typename std::remove_cvref_t<Data>::policy_or_iter_space_t>,
        typename std::remove_cvref_t<Data>::reducer_t,
        typename std::remove_cvref_t<Data>::properties_t
    >;

    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelReduceSender<parallel_reduce_t, Sndr, data_t<Env, Data, Sndr>>;

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
            {.data = std::forward<Data>(data).transform_policy_or_iter_space(
                 [&]<typename T>(T&& pol) { // NOLINT(cppcoreguidelines-missing-std-forward)
                     return Impl::make_policy(schd.state->exec, std::forward<T>(pol));
                 })},
            std::forward<Sndr>(sndr)};
    }

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, Kokkos::Execution::parallel_reduce_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<parallel_reduce_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_REDUCE_HPP
