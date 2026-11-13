#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/diagnostics.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <Kokkos::ExecutionSpace Exec, typename Label, typename Functor>
struct ThenClosure {
    using execution_space = Exec;

    void submit() const & {
        Kokkos::single(Impl::as_convertible_to_string(label), this->get_policy(), functor);
    }

    //! @todo Implement policy update for the single policy in kokkos.
    [[nodiscard]]
    auto get_policy() const & noexcept {
        return Kokkos::SinglePolicy<execution_space>{state->exec};
    }

    const Impl::State<execution_space>* state;
    Label label;
    Functor functor;
};

template <typename Tag, stdexec::sender Sndr, Kokkos::ExecutionSpace Exec, typename Label, typename Functor>
struct ThenSender {
    using sender_concept = stdexec::sender_tag;

    using execution_space = Exec;
    using closure_t = ThenClosure<execution_space, Label, Functor>;

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
    template <typename Env, typename Functor, typename Sndr>
    struct Transformer {
        using execution_space = Impl::exec_of_t<Sndr, Env>;

        using sndr_t = ParallelForSender<
            stdexec::then_t,
            Sndr,
            PforData... execution_space,
            std::string_view,
            std::remove_cvref_t<Functor>
        >;

        auto operator()(const Env& env, std::remove_reference_t<Functor>& functor, std::remove_reference_t<Sndr>& sndr)
            const STDEXEC_AUTO_RETURN(
                sndr_t{
                    {.state = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                                  .state,
                     .label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(execution_space, ": then"),
                     .data = {.iter_space = Kokkos::range_slice<integral_cst(0), integral_cst(1)>{}... = std::forward<Functor>(functor)},
                    std::forward<Sndr>(sndr)
        })
    };

    template <typename Env, typename Functor, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        stdexec::then_t,
        Functor&& functor, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr        // NOLINT(cppcoreguidelines-missing-std-forward)
    ) const STDEXEC_AUTO_RETURN(Transformer<Env, Functor, Sndr>{}(env, functor, sndr))

    template <typename Env, typename Functor, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::then_t, Functor&&, Sndr&&) const noexcept {
            return no_execution_space_scheduler_in_env<stdexec::then_t, Sndr, Env>();
    }
    };

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
