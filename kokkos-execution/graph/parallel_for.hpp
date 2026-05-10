#ifndef KOKKOS_EXECUTION_GRAPH_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_GRAPH_PARALLEL_FOR_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/parallel_for.hpp"

namespace Kokkos::Execution::GraphImpl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForClosure {
    using execution_space = typename ExecPolicy::execution_space;
    using device_handle_t = Kokkos::Impl::DeviceHandle<execution_space>;
    using node_props_t = typename Kokkos::Impl::NodeCtorProps<Label, device_handle_t>::uniform_type;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>().then_parallel_for(
        std::declval<node_props_t&&>(),
        std::declval<ExecPolicy&&>(),
        std::declval<Functor&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && noexcept(false) -> node_t<Predecessor> {
        auto device_handle = Kokkos::Impl::get_property<device_handle_t>(node_props);
        auto node = predecessor.then_parallel_for(
            std::move(node_props), std::forward<ExecPolicy>(policy), std::forward<Functor>(functor));
        KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING("parallel_for", node, predecessor)
        graph_add_node_event(predecessor, node, device_handle);
        return node;
    }
    node_props_t node_props;
    Functor functor; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    ExecPolicy policy;
};

template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelForClosure<Label, Functor, ExecPolicy>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelForSender, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
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

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(
        const Env& env,
        Kokkos::Execution::parallel_for_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            auto& [label, functor, policy] = data;

            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

            return trnsfrmd_sndr_t<Env, Data, Sndr>{
                .clsr =
                    {.node_props = Kokkos::Experimental::node_props(
                         stdexec::__forward_like<Data>(label),
                           Kokkos::Experimental::get_device_handle(schd.state->exec)),
                           .functor = stdexec::__forward_like<Data>(functor),
                           .policy = stdexec::__forward_like<Data>(policy)},
                .sndr = std::forward<Sndr>(sndr)
            };
        } else {
            return no_graph_scheduler_in_env<parallel_for_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace stdexec::__detail {
template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
extern __mtype<Kokkos::Execution::GraphImpl::ParallelForSender<__demangle_t<Sndr>, Label, Functor, ExecPolicy>>
    __demangle_v<Kokkos::Execution::GraphImpl::ParallelForSender<Sndr, Label, Functor, ExecPolicy>>;
} // namespace stdexec::__detail
// NOLINTEND(bugprone-reserved-identifier)

#endif // KOKKOS_EXECUTION_GRAPH_PARALLEL_FOR_HPP
