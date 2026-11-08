#ifndef KOKKOS_EXECUTION_GRAPH_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_GRAPH_PARALLEL_REDUCE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/node_props.hpp"
#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/make_policy.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/impl/state.hpp"
#include "kokkos-execution/parallel_reduce.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, Impl::parallel_reduce_data Data>
struct ParallelReduceClosure {
    using execution_space = Exec;

    using device_handle_t = Kokkos::Impl::DeviceHandle<execution_space>;

    using node_props_t = make_node_props_t<Data, execution_space>;
    using policy_t =
        Impl::make_policy_result_t<Impl::make_policy_t<execution_space>, const typename Data::policy_or_iter_space_t&>;
    using functor_t = typename Data::functor_t;
    using reducer_t = typename Data::reducer_t;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>().then_parallel_reduce(
        std::declval<node_props_t&&>(),
        std::declval<policy_t&&>(),
        std::declval<functor_t&&>(),
        std::declval<reducer_t&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && -> node_t<Predecessor> {
        node_props_t node_props = std::move(*this).get_node_props();
        policy_t policy = this->get_policy();
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        auto label = get_label_or_display_string_of<functor_t>(node_props); // NOLINT(misc-const-correctness)
#endif
        auto node = predecessor.then_parallel_reduce(
            std::move(node_props),
            std::move(policy),
            stdexec::__forward_like<Data>(data.functor),
            stdexec::__forward_like<Data>(data.reducer));
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING(label, node, predecessor)
        graph_add_node_event(std::move(label), predecessor, node);
#endif
        return node;
    }

    template <typename Self>
    [[nodiscard]]
    STDEXEC_EXPLICIT_THIS_BEGIN(auto get_node_props)(this Self&& self) // NOLINT(cppcoreguidelines-missing-std-forward)
        STDEXEC_AUTO_RETURN(make_node_props<stdexec::__copy_cvref_t<Self, Data>>(self.get_device_handle(), self.data))
    STDEXEC_EXPLICIT_THIS_END(get_node_props)

    [[nodiscard]]
    auto get_policy() const & noexcept -> policy_t {
        return Impl::make_policy<execution_space>(data.policy_or_iter_space);
    }

    [[nodiscard]]
    auto get_device_handle() const & noexcept -> device_handle_t {
        return Kokkos::Experimental::get_device_handle(state->exec);
    }

    const Impl::State<execution_space>* state;
    Data data;
};

template <stdexec::sender Sndr, Kokkos::ExecutionSpace Exec, Impl::parallel_reduce_data Data>
struct ParallelReduceSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ParallelReduceClosure<Exec, Data>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelReduceSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<Kokkos::Execution::parallel_reduce_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelReduceSender<Sndr, Impl::exec_of_t<Sndr, Env>, std::remove_cvref_t<Data>>;

    template <typename Env, typename Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
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
        return no_graph_scheduler_in_env<parallel_reduce_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_PARALLEL_REDUCE_HPP
