#ifndef KOKKOS_EXECUTION_GRAPH_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_GRAPH_CONTINUES_ON_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/graph/graph_fwd.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/type_traits.hpp"

namespace Kokkos::Execution::GraphImpl {

//! Receiver for @c stdexec::continues_on.
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct ContinuesOnReceiver : public stdexec::__receiver_adaptor<ContinuesOnReceiver<Exec, Rcvr>, Rcvr> {
    using receiver_concept = stdexec::receiver_tag;

    using execution_space = Exec;

    using base_t = stdexec::__receiver_adaptor<ContinuesOnReceiver<Exec, Rcvr>, Rcvr>;

    explicit ContinuesOnReceiver(Rcvr rcvr)
        : base_t(std::move(rcvr)) {
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, this->base())
};

//! Operation state for @c stdexec::continues_on.
template <Kokkos::ExecutionSpace Exec, stdexec::sender Sndr, stdexec::receiver Rcvr>
struct ContinuesOnOpState {
    using operation_state_concept = stdexec::operation_state_tag;

    using rcvr_t = ContinuesOnReceiver<Exec, Rcvr>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    inner_opstate_t inner_opstate;

    ContinuesOnOpState(Sndr&& sndr, Rcvr rcvr) noexcept( // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{std::move(rcvr)})) {
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

//! Sender for @c stdexec::continues_on.
template <typename Schd, typename Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_tag;

    using execution_space = typename std::remove_cvref_t<Schd>::execution_space;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    /**
     * The selected graph composition will be @ref GraphComposition::Attach only if:
     *  1. The predecessor completion domain is @ref Kokkos::Execution::GraphImpl::Domain.
     *  2. The execution space type is the same as @ref execution_space.
     *
     * Otherwise, @ref GraphComposition::Create is selected.
     */
    template <typename Self, typename Rcvr>
    static consteval auto graph_composition_policy() noexcept {
        if constexpr (
            std::same_as<
                stdexec::__completion_domain_of_t<
                    stdexec::set_value_t,
                    KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr),
                    stdexec::env_of_t<Rcvr>
                >,
                Domain
            >) {
            return std::conditional_t<
                std::same_as<
                    Impl::exec_of_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), stdexec::env_of_t<Rcvr>>,
                    execution_space
                >,
                GraphComposition::Attach,
                GraphComposition::Create
            >{};
        } else {
            return GraphComposition::Create{};
        }
    }

    template <typename Self, typename Rcvr>
    using graph_composition_policy_t = decltype(graph_composition_policy<Self, Rcvr>());

    template <typename Self, typename Rcvr>
    using connect_result_t = std::conditional_t<
        std::same_as<graph_composition_policy_t<Self, Rcvr>, GraphComposition::Attach>,
        stdexec::connect_result_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr>,
        ContinuesOnOpState<execution_space, KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr>
    >;

    template <typename Self, typename Rcvr>
    static constexpr bool is_connect_nothrow =
        std::same_as<graph_composition_policy_t<Self, Rcvr>, GraphComposition::Attach>
            ? stdexec::__nothrow_connectable<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr>
            : std::is_nothrow_constructible_v<
                  ContinuesOnOpState<execution_space, KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr>,
                  KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr),
                  Rcvr&&
              >;

    template <stdexec::__decays_to<ContinuesOnSender> Self, stdexec::receiver Rcvr>
    STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(is_connect_nothrow<Self, Rcvr>) -> connect_result_t<Self, Rcvr> {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << "'continues_on' sender is " << Kokkos::Impl::TypeInfo<stdexec::__demangle_t<Sndr>>::name();
#endif
        if constexpr (std::same_as<graph_composition_policy_t<Self, Rcvr>, GraphComposition::Attach>) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_DEBUG << "'continues_on' graph composition policy is ATTACH.";
#endif
            return stdexec::connect(std::forward<Self>(self).sndr, std::move(rcvr));
        } else {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_DEBUG << "'continues_on' graph composition policy is CREATE.";
#endif
            return {std::forward<Self>(self).sndr, std::move(rcvr)};
        }
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Schd schd; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::continues_on_t> {
    template <typename Env, stdexec::scheduler Schd, stdexec::sender Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Schd, Sndr>, Schd&&, Sndr&&>)
            -> ContinuesOnSender<Schd, Sndr> {
        static_assert(stdexec::__is_instance_of<std::remove_cvref_t<Schd>, Scheduler>);
        return {.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_CONTINUES_ON_HPP
