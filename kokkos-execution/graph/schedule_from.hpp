#ifndef KOKKOS_EXECUTION_GRAPH_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_GRAPH_SCHEDULE_FROM_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/get_graph.hpp"
#include "kokkos-execution/graph/get_node.hpp"
#include "kokkos-execution/graph/graph_fwd.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/sync_wait.hpp"

namespace Kokkos::Execution::GraphImpl {

enum class Synchronization : std::uint8_t {
    COMPILE_TIME_NO,
    COMPILE_TIME_YES,
    RUNTIME
};

std::ostream& operator<<(std::ostream& out, const Synchronization synchronization) {
    switch (synchronization) {
    case Synchronization::COMPILE_TIME_NO:
        return out << "compile-time NO";
    case Synchronization::COMPILE_TIME_YES:
        return out << "compile-time YES";
    case Synchronization::RUNTIME:
        return out << "runtime";
    default:
        Kokkos::abort("unreachable");
    }
}

//! Operation state for @c stdexec::schedule_from that waits for the graph submission to complete.
template <Synchronization synchronization, typename Exec, typename Sndr, typename Rcvr>
struct ScheduleFromOpState {
    using operation_state_concept = stdexec::operation_state_tag;

    static_assert(synchronization == Synchronization::COMPILE_TIME_YES || synchronization == Synchronization::RUNTIME);

    using receiver_t = Rcvr;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, Impl::Receiver<ScheduleFromOpState>>;

    Rcvr rcvr;
    inner_opstate_t inner_opstate;

    //! @todo Better @c noexcept.
    ScheduleFromOpState(Sndr&& sndr, Rcvr rcvr_) // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        noexcept(false)
        : rcvr(std::move(rcvr_))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), Impl::Receiver<ScheduleFromOpState>{this})) {
    }

    void propagate_completion_signal(stdexec::set_value_t) noexcept {
        const auto& graph = inner_opstate.query(get_graph);
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Synchronizing graph " << get_graph_impl_ptr(graph.root_node()) << " on "
                  << Kokkos::Tools::Experimental::device_id(graph.get_device_handle().m_exec) << '.';
#endif
        const bool requires_synchronization = [&]() {
            if constexpr (synchronization == Synchronization::COMPILE_TIME_YES) {
                return true;
            } else {
                static_assert(stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, Impl::get_exec_t>);
                static_assert(std::same_as<
                              typename std::remove_cvref_t<
                                  stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, Impl::get_exec_t>
                              >::execution_space,
                              Exec
                >);
                return Impl::get_exec(stdexec::get_env(rcvr)).get() != graph.get_device_handle().m_exec;
            }
        }();
        if (requires_synchronization)
            graph.get_device_handle()
                .m_exec.fence(std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<Exec>::name()));
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Tag, typename... Args>
    void propagate_completion_signal(Tag, Args&&... args) noexcept {
        Tag{}(std::move(rcvr), std::forward<Args>(args)...);
    }

    void submit() & noexcept {
        this->propagate_completion_signal(stdexec::set_value);
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }

    const auto& query(get_node_t) const noexcept {
        return inner_opstate.query(get_node);
    }

    const auto& query(get_graph_t) const noexcept {
        return inner_opstate.query(get_graph);
    }
};

/**
 * @brief Sender for @c stdexec::schedule_from.
 *
 * It must handle three cases:
 *  1. If the downstream receiver is a @ref Kokkos::Execution::GraphImpl::ContinuesOnReceiver:
 *      * if its execution space type is the same, it will attach to the current graph (no synchronization needed)
 *      * otherwise, it will create a new graph (synchronization needed)
 *  2. If the downstream receiver is queryable for @ref Kokkos::Execution::Impl::get_exec_t:
 *      * if its execution space type is the same, the synchronization decision happens at runtime
 *      * otherwise, it always needs to synchronize
 *  3. Always synchronize.
 *
 * When the synchronization is known to happen at compile-time, use @ref Synchronization::COMPILE_TIME_YES.
 * When the synchronization is known not to happen at compile-time, use @ref Synchronization::COMPILE_TIME_NO.
 * When the decision is runtime, use @ref Synchronization::RUNTIME.
 */
template <Kokkos::ExecutionSpace Exec, typename Sndr>
struct ScheduleFromSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <typename Rcvr>
    static constexpr Synchronization synchronization = []() {
        if constexpr (Impl::supports_submitted_order_on<Rcvr>) {
            return Synchronization::COMPILE_TIME_NO;
        }
        if constexpr (stdexec::__is_instance_of<Rcvr, ContinuesOnReceiver>) {
            return std::same_as<Exec, typename Rcvr::execution_space> ? Synchronization::COMPILE_TIME_NO
                                                                      : Synchronization::COMPILE_TIME_YES;
        }
        if constexpr (stdexec::__is_instance_of<Rcvr, Impl::SyncWait::Receiver>)
            return Synchronization::COMPILE_TIME_NO;
        if constexpr (stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, Impl::get_exec_t>) {
            if constexpr (
                std::same_as<
                    typename std::remove_cvref_t<
                        stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, Impl::get_exec_t>
                    >::execution_space,
                    Exec
                >) {
                return Synchronization::RUNTIME;
            }
        }
        return Synchronization::COMPILE_TIME_YES;
    }();

    template <typename Sender, typename Rcvr>
    using my_connect_result_t = std::conditional_t<
        synchronization<Rcvr> == Synchronization::COMPILE_TIME_NO,
        stdexec::connect_result_t<Sender, Rcvr>,
        ScheduleFromOpState<synchronization<Rcvr>, Exec, Sender, Rcvr>
    >;

    template <stdexec::__decays_to<ScheduleFromSender> Self, stdexec::receiver Rcvr>
    STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(stdexec::__nothrow_connectable<stdexec::__copy_cvref_t<Self, Sndr>, Rcvr&&>)
            -> my_connect_result_t<stdexec::__copy_cvref_t<Self, Sndr>, Rcvr> {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << "'schedule_from' synchronization is " << synchronization<Rcvr> << '.';
        PLOG_INFO << "Rcvr type is " << Kokkos::Impl::TypeInfo<Rcvr>::name();
#endif
        if constexpr (synchronization<Rcvr> == Synchronization::COMPILE_TIME_NO) {
            return stdexec::connect(stdexec::__forward_like<Self>(self.sndr), std::move(rcvr));
        } else {
            return ScheduleFromOpState<synchronization<Rcvr>, Exec, stdexec::__copy_cvref_t<Self, Sndr>, Rcvr>(
                stdexec::__forward_like<Self>(self.sndr), std::move(rcvr));
        }
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::schedule_from_t> {
    template <typename Env, stdexec::sender Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ScheduleFromSender<Impl::exec_of_t<Sndr, Env>, Sndr>, Sndr&&>) {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            return ScheduleFromSender<Impl::exec_of_t<Sndr, Env>, Sndr>{.sndr = std::forward<Sndr>(sndr)};
        } else {
            return no_graph_scheduler_in_env<stdexec::schedule_from_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_SCHEDULE_FROM_HPP
