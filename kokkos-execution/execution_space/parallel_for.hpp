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

template <stdexec::__is_instance_of<Impl::ParallelForData> Data>
struct ParallelForClosure {
    using policy_t = typename Data::policy_t;
    using execution_space = typename policy_t::execution_space;

    Data data;

    void execute() const & {
        if constexpr (Impl::labeled<Data>) {
            Kokkos::parallel_for(data.label, data.policy, data.functor);
        } else {
            Kokkos::parallel_for(data.policy, data.functor);
        }
    }

    const policy_t& get_policy() const & noexcept {
        return data.policy;
    }
};

template <stdexec::sender Sndr, stdexec::__is_instance_of<Impl::ParallelForData> Data>
struct ParallelForSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelForClosure<Data>;
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
    using trnsfrmd_sndr_t = ParallelForSender<Sndr, std::remove_cvref_t<Data>>;

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env& env, Kokkos::Execution::parallel_for_t, Data&& data, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 parallel_for_t,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        using sndr_t = trnsfrmd_sndr_t<Env, Data, Sndr>;

        //! Only the execution space instance, not its type, can be bound lately.
        static_assert(
            std::same_as<exec_of_t<Sndr, Env>, typename sndr_t::closure_t::execution_space>,
            "The policy's execution space type must be the same as the completion scheduler's execution space type.");

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        typename std::remove_cvref_t<Data>::policy_t policy(
            Kokkos::Impl::PolicyUpdate{}, stdexec::__forward_like<Data>(data.policy), schd.state->exec);

        return sndr_t{
            parallel_for_t{},
            make_closure<typename sndr_t::closure_t>(std::forward<Data>(data), std::move(policy)),
            std::forward<Sndr>(sndr)};
    }

   private:
    template <typename Clsr, typename Data, typename Policy>
    static auto make_closure(Data&& data, Policy&& policy) -> Clsr { // NOLINT(cppcoreguidelines-missing-std-forward)
        if constexpr (Impl::labeled<std::remove_cvref_t<Data>>) {
            return Clsr{
                Impl::Label{stdexec::__forward_like<Data>(data.label)},
                stdexec::__forward_like<Data>(data.functor),
                std::forward<Policy>(policy)};
        } else {
            return Clsr{stdexec::__forward_like<Data>(data.functor), std::forward<Policy>(policy)};
        }
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#if !STDEXEC_HAS_BUILTIN(__builtin_structured_binding_size)
namespace stdexec {

//! See also https://cor3ntin.github.io/posts/clang21/#__builtin_structured_binding_size.
template <stdexec::sender Sndr, typename Data>
inline constexpr auto __structured_binding_size_v< // NOLINT(bugprone-reserved-identifier)
    Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<Sndr, Data>
> = 3;

} // namespace stdexec
#endif

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PARALLEL_FOR_HPP
