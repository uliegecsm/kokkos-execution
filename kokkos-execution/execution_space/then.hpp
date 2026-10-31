#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/diagnostics.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ThenClosure {
    static_assert(Kokkos::Impl::is_specialization_of_v<ExecPolicy, SinglePolicy>);

    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    void submit() const & {
        Kokkos::single(Impl::as_convertible_to_string(label), policy, functor);
    }

    [[nodiscard]]
    const policy_t& get_policy() const & noexcept {
        return policy;
    }

    Label label;
    Functor functor;
    policy_t policy;
};

//! Deduction guide to store by-value.
template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
ThenClosure(Label, Functor, ExecPolicy) -> ThenClosure<Label, Functor, ExecPolicy>;

template <typename Tag, stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ThenSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ThenClosure<Label, Functor, ExecPolicy>;
    using execution_space = typename closure_t::execution_space;

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ThenSender, Sndr, stdexec::set_error_t(std::exception_ptr))

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
struct TransformSenderFor<stdexec::then_t> {
    template <typename Sndr, typename Env>
    using policy_t = Impl::make_policy_t<Impl::exec_of_t<Sndr, Env>, Kokkos::dims<0>>;

    template <typename Env, typename Functor, typename Sndr>
    using trnsfrmd_sndr_t =
        ThenSender<stdexec::then_t, Sndr, std::string_view, std::remove_cvref_t<Functor>, policy_t<Sndr, Env>>;

    template <typename Env, typename Functor, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(const Env& env, stdexec::then_t, Functor&& functor, Sndr&& sndr) const STDEXEC_AUTO_RETURN(
        trnsfrmd_sndr_t<Env, Functor, Sndr>{
            {.label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL((Impl::exec_of_t<Sndr, Env>), ": then"),
             .functor = std::forward<Functor>(functor),
             .policy = Impl::make_policy(
                 stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env).state->exec,
             Kokkos::dims<0>{})},
            std::forward<Sndr>(sndr)
    })

    template <typename Env, typename Functor, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::then_t, Functor&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<stdexec::then_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
