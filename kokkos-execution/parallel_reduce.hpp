#ifndef KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/iteration_space.hpp"
#include "kokkos-execution/impl/label.hpp"
#include "kokkos-execution/impl/macros.hpp"
#include "kokkos-execution/impl/properties.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <typename Data>
concept parallel_reduce_data = requires {
    typename Data::functor_t;
    typename Data::policy_or_iter_space_t;
    typename Data::reducer_t;
    typename Data::properties_t;

    requires execution_policy_or_iteration_space<typename Data::policy_or_iter_space_t>;
    requires stdexec::__is_instance_of<typename Data::properties_t, properties>;
    requires requires(const Data& data) {
        { data.functor } -> std::same_as<const typename Data::functor_t&>;
        { data.policy_or_iter_space } -> std::same_as<const typename Data::policy_or_iter_space_t&>;
        { data.reducer } -> std::same_as<const typename Data::reducer_t&>;
    };
};

template <
    typename Functor,
    execution_policy_or_iteration_space ExecPolicyOrIterSpace,
    typename Reducer,
    typename... Properties
>
struct ParallelReduceData;

template <stdexec::sender Sndr, parallel_reduce_data Data>
struct ParallelReduceSender;

} // namespace Impl

//! Custom algorithm for the @c Kokkos::parallel_reduce construct.
struct parallel_reduce_t {
    template <typename Label, typename Functor, typename ExecPolicy, typename Reducer>
    requires(Impl::label<std::remove_cvref_t<Label>> && Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>)
    constexpr auto
        operator()(Label&& label, ExecPolicy&& policy, Functor&& functor, Reducer&& reducer) const STDEXEC_AUTO_RETURN(
            stdexec::__closure(
                *this,
                std::forward<ExecPolicy>(policy),
                std::forward<Functor>(functor),
                std::forward<Reducer>(reducer),
                Impl::Label{std::forward<Label>(label)}))

    template <typename ExecPolicyOrIterSpace, typename Functor, typename Reducer>
    requires Impl::execution_policy_or_iteration_space<std::remove_cvref_t<ExecPolicyOrIterSpace>>
    constexpr auto operator()(ExecPolicyOrIterSpace&& policy_or_iter_space, Functor&& functor, Reducer&& reducer) const STDEXEC_AUTO_RETURN(
        stdexec::__closure(
            *this,
            std::forward<ExecPolicyOrIterSpace>(policy_or_iter_space),
            std::forward<Functor>(functor),
            std::forward<Reducer>(reducer)))

    template <typename Label, typename Functor, std::integral T, typename Reducer>
    requires Impl::label<std::remove_cvref_t<Label>>
    constexpr auto
        operator()(Label&& label, const T work_count, Functor&& functor, Reducer&& reducer) const STDEXEC_AUTO_RETURN(
            stdexec::__closure(
                *this,
                Impl::make_iter_space(work_count),
                std::forward<Functor>(functor),
                std::forward<Reducer>(reducer),
                Impl::Label{std::forward<Label>(label)}))

    template <typename Functor, std::integral T, typename Reducer>
    constexpr auto operator()(const T work_count, Functor&& functor, Reducer&& reducer) const STDEXEC_AUTO_RETURN(
        stdexec::__closure(
            *this, Impl::make_iter_space(work_count), std::forward<Functor>(functor), std::forward<Reducer>(reducer)))

    template <typename Functor, typename ExecPolicyOrIterSpace, typename Reducer, typename... Properties>
    using parallel_reduce_data_t = Impl::ParallelReduceData<
        std::remove_cvref_t<Functor>,
        std::remove_cvref_t<ExecPolicyOrIterSpace>,
        std::remove_cvref_t<Reducer>,
        std::remove_cvref_t<Properties>...
    >;

    template <
        stdexec::sender Sndr,
        typename Functor,
        typename ExecPolicyOrIterSpace,
        typename Reducer,
        typename... Properties
    >
    requires Impl::execution_policy_or_iteration_space<std::remove_cvref_t<ExecPolicyOrIterSpace>>
    constexpr auto operator()(
        Sndr&& sndr,
        ExecPolicyOrIterSpace&& policy_or_iter_space,
        Functor&& functor,
        Reducer&& reducer,
        Properties&&... properties) const
        KOKKOS_EXECUTION_AUTO_RETURN_AS(
            (Impl::ParallelReduceSender<
                Sndr,
                parallel_reduce_data_t<Functor, ExecPolicyOrIterSpace, Reducer, Properties...>
            >),
            Impl::ParallelReduceSender<
                Sndr,
                parallel_reduce_data_t<Functor, ExecPolicyOrIterSpace, Reducer, Properties...>
            >{
                parallel_reduce_t{},
                parallel_reduce_data_t<Functor, ExecPolicyOrIterSpace, Reducer, Properties...>{
                                  std::forward<Properties>(properties)...,
                                  std::forward<Functor>(functor),
                                  std::forward<ExecPolicyOrIterSpace>(policy_or_iter_space),
                                  std::forward<Reducer>(reducer)},
                std::forward<Sndr>(sndr)
    })
};

namespace Impl {

template <
    typename Functor,
    execution_policy_or_iteration_space ExecPolicyOrIterSpace,
    typename Reducer,
    typename... Properties
>
struct ParallelReduceData : Properties... {
    using functor_t = Functor;
    using policy_or_iter_space_t = ExecPolicyOrIterSpace;
    using reducer_t = Reducer;
    using properties_t = properties<Properties...>;

    functor_t functor;
    policy_or_iter_space_t policy_or_iter_space;
    reducer_t reducer;

    //! Transform @ref policy_or_iter_space, keeping other data unchanged.
    template <typename Self, typename UnaryOp>
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(auto transform_policy_or_iter_space)(
        this Self&& self, // NOLINT(cppcoreguidelines-missing-std-forward)
        UnaryOp&& unary_op)
        STDEXEC_AUTO_RETURN(
            ParallelReduceData<
                functor_t,
                std::remove_cvref_t<
                    std::invoke_result_t<UnaryOp, decltype(stdexec::__forward_like<Self>(self.policy_or_iter_space))>
                >,
                reducer_t,
                Properties...
            >{stdexec::__forward_like<Self>(detail::as_base<Properties>(self))...,
              stdexec::__forward_like<Self>(self.functor),
              std::forward<UnaryOp>(unary_op)(stdexec::__forward_like<Self>(self.policy_or_iter_space)),
              stdexec::__forward_like<Self>(self.reducer)})

    STDEXEC_EXPLICIT_THIS_END(transform_policy_or_iter_space)
};

//! @name Get the @ref ParallelReduceData type from its components.
///@{
template <typename Functor, typename ExecPolicyOrIterSpace, typename Reducer, typename... Properties>
struct parallel_reduce_data_from {
    using type = ParallelReduceData<Functor, ExecPolicyOrIterSpace, Reducer, Properties...>;
};

template <typename Functor, typename ExecPolicyOrIterSpace, typename Reducer, typename... Properties>
struct parallel_reduce_data_from<Functor, ExecPolicyOrIterSpace, Reducer, properties<Properties...>> {
    using type = ParallelReduceData<Functor, ExecPolicyOrIterSpace, Reducer, Properties...>;
};

template <typename Functor, typename ExecPolicyOrIterSpace, typename Reducer, typename... Properties>
using parallel_reduce_data_from_t =
    typename parallel_reduce_data_from<Functor, ExecPolicyOrIterSpace, Reducer, Properties...>::type;

///@}

template <stdexec::sender Sndr, Impl::parallel_reduce_data Data>
struct ParallelReduceSender : stdexec::__tuple<parallel_reduce_t, Data, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = Data;                       // NOLINT(bugprone-reserved-identifier)
    using __children_t = stdexec::__mlist<Sndr>; // NOLINT(bugprone-reserved-identifier)
    ///@}

    using base_t = stdexec::__tuple<parallel_reduce_t, __data_t, Sndr>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelReduceSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr) && = delete;

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr) const & = delete;

    static constexpr size_t idx_sndr = 2;
    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(
        Sndr,
        stdexec::__get<idx_sndr>(static_cast<const base_t&>(*this)))
};

} // namespace Impl

inline constexpr parallel_reduce_t parallel_reduce{};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP
