#ifndef KOKKOS_EXECUTION_GRAPH_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_GRAPH_PARALLEL_REDUCE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/node_props.hpp"
#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/make_policy.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/parallel_reduce.hpp"

namespace Kokkos::Execution::GraphImpl {

template <
    typename Label,
    typename DeviceHandle,
    typename Functor,
    Impl::execution_policy_or_iteration_space ExecPolicyOrIterSpace,
    typename Reducer
>
struct ParallelReduceClosure {
    using device_handle_t = DeviceHandle;
    using execution_space = exec_of_t<device_handle_t>;
    using node_props_t = typename Kokkos::Impl::NodeCtorProps<device_handle_t, std::string>::uniform_type;
    using policy_t = Impl::make_policy_t<execution_space, ExecPolicyOrIterSpace&&>;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>().then_parallel_reduce(
        std::declval<node_props_t&&>(),
        std::declval<policy_t&&>(),
        std::declval<Functor&&>(),
        std::declval<Reducer&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && noexcept(false) -> node_t<Predecessor> {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        auto label_copy = std::string(label);
        const auto device_handle_copy = device_handle;
#endif
        auto node = predecessor.then_parallel_reduce(
            Kokkos::Experimental::node_props(std::move(device_handle), std::string(std::move(label))),
            Impl::make_policy(execution_space{}, std::move(policy_or_iter_space)),
            std::move(functor),
            std::move(reducer));
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING(label_copy, node, predecessor)
        graph_add_node_event(std::move(label_copy), predecessor, node, device_handle_copy);
#endif
        return node;
    }

    Label label;
    device_handle_t device_handle;
    Functor functor;
    ExecPolicyOrIterSpace policy_or_iter_space;
    Reducer reducer;
};

template <
    stdexec::sender Sndr,
    typename Label,
    typename DeviceHandle,
    typename Functor,
    Impl::execution_policy_or_iteration_space ExecPolicyOrIterSpace,
    typename Reducer
>
struct ParallelReduceSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelReduceClosure<Label, DeviceHandle, Functor, ExecPolicyOrIterSpace, Reducer>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelReduceSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<Kokkos::Execution::parallel_reduce_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelReduceSender<
        Sndr,
        typename std::remove_cvref_t<Data>::label_t,
        Kokkos::Impl::DeviceHandle<Impl::exec_of_t<Sndr, Env>>,
        typename std::remove_cvref_t<Data>::functor_t,
        typename std::remove_cvref_t<Data>::policy_or_iter_space_t,
        typename std::remove_cvref_t<Data>::reducer_t
    >;

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, Kokkos::Execution::parallel_reduce_t, Data&&, Sndr&&) const noexcept {
        return no_graph_scheduler_in_env<parallel_reduce_t, Sndr, Env>();
    }

    template <typename Env, typename Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        Kokkos::Execution::parallel_reduce_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        STDEXEC_AUTO_RETURN(
            trnsfrmd_sndr_t<Env, Data, Sndr>{
                .clsr =
                    {.label = stdexec::__forward_like<Data>(data.label),
                           .device_handle =
                         stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                             .state->get_device_handle(),
                           .functor = stdexec::__forward_like<Data>(data.functor),
                           .policy_or_iter_space = stdexec::__forward_like<Data>(data.policy_or_iter_space),
                           .reducer = stdexec::__forward_like<Data>(data.reducer)},
                .sndr = std::forward<Sndr>(sndr)
    })
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_PARALLEL_REDUCE_HPP
