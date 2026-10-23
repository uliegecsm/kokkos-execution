#ifndef KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/iteration_space.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <
    stdexec::sender Sndr,
    typename Label,
    typename Functor,
    execution_policy_or_iteration_space ExecPolicyOrIterSpace,
    typename Reducer
>
struct ParallelReduceSender;

template <typename Label, typename Functor, execution_policy_or_iteration_space ExecPolicyOrIterSpace, typename Reducer>
struct ParallelReduceData;

} // namespace Impl

//! Custom algorithm for the @c Kokkos::parallel_reduce construct.
struct parallel_reduce_t {
    template <typename Functor, typename ExecPolicy, typename Reducer>
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
    constexpr auto operator()(std::string label, ExecPolicy&& policy, Functor&& functor, Reducer&& reducer) const
        noexcept(stdexec::__nothrow_decay_copyable<Functor, ExecPolicy, Reducer>) {
        return stdexec::__closure(
            *this,
            std::move(label),
            std::forward<ExecPolicy>(policy),
            std::forward<Functor>(functor),
            std::forward<Reducer>(reducer));
    }

    template <typename IterSpace, typename Functor, typename Reducer>
    requires Impl::iteration_space<std::remove_cvref_t<IterSpace>>
    constexpr auto operator()(IterSpace&& iter_space, Functor&& functor, Reducer&& reducer) const {
        return stdexec::__closure(
            *this,
            "",
            std::forward<IterSpace>(iter_space),
            std::forward<Functor>(functor),
            std::forward<Reducer>(reducer));
    }

    template <typename Functor, typename ExecPolicy, typename Reducer>
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
    constexpr auto operator()(ExecPolicy&& policy, Functor&& functor, Reducer&& reducer) const {
        return this->operator()(
            "", std::forward<ExecPolicy>(policy), std::forward<Functor>(functor), std::forward<Reducer>(reducer));
    }

    template <typename Functor, std::integral T, typename Reducer>
    constexpr auto operator()(std::string label, const T work_count, Functor&& functor, Reducer&& reducer) const {
        return stdexec::__closure(
            *this,
            std::move(label),
            Kokkos::dims<1, T>{work_count},
            std::forward<Functor>(functor),
            std::forward<Reducer>(reducer));
    }

    template <typename Functor, std::integral T, typename Reducer>
    constexpr auto operator()(const T work_count, Functor&& functor, Reducer&& reducer) const {
        return this->operator()("", work_count, std::forward<Functor>(functor), std::forward<Reducer>(reducer));
    }

    template <stdexec::sender Sndr, typename Functor, typename ExecPolicyOrIterSpace, typename Reducer>
    requires Impl::execution_policy_or_iteration_space<std::remove_cvref_t<ExecPolicyOrIterSpace>>
    constexpr auto operator()(
        Sndr&& sndr,
        std::string label,
        ExecPolicyOrIterSpace&& policy_or_iter_space,
        Functor&& functor,
        Reducer&& reducer) const
        noexcept(stdexec::__nothrow_decay_copyable<Sndr, Functor, ExecPolicyOrIterSpace, Reducer>)
            -> Impl::ParallelReduceSender<
                Sndr,
                std::string,
                std::remove_cvref_t<Functor>,
                std::remove_cvref_t<ExecPolicyOrIterSpace>,
                std::remove_cvref_t<Reducer>
            > {
        return {
            {parallel_reduce_t{},
             Impl::ParallelReduceData{
                 std::move(label),
                 std::forward<Functor>(functor),
                 std::forward<ExecPolicyOrIterSpace>(policy_or_iter_space),
                 std::forward<Reducer>(reducer)},
             std::forward<Sndr>(sndr)}
        };
    }
};

namespace Impl {

template <typename Label, typename Functor, execution_policy_or_iteration_space ExecPolicyOrIterSpace, typename Reducer>
struct ParallelReduceData {
    using label_t = Label;
    using functor_t = Functor;
    using policy_or_iter_space_t = ExecPolicyOrIterSpace;
    using reducer_t = Reducer;

    label_t label;
    functor_t functor;
    policy_or_iter_space_t policy_or_iter_space;
    reducer_t reducer;
};

//! Deduction guide to store by-value.
template <typename Label, typename Functor, execution_policy_or_iteration_space ExecPolicyOrIterSpace, typename Reducer>
ParallelReduceData(Label, Functor, ExecPolicyOrIterSpace, Reducer)
    -> ParallelReduceData<Label, Functor, ExecPolicyOrIterSpace, Reducer>;

template <
    stdexec::sender Sndr,
    typename Label,
    typename Functor,
    execution_policy_or_iteration_space ExecPolicyOrIterSpace,
    typename Reducer
>
struct ParallelReduceSender
    : stdexec::__tuple<parallel_reduce_t, ParallelReduceData<Label, Functor, ExecPolicyOrIterSpace, Reducer>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = // NOLINT(bugprone-reserved-identifier)
        ParallelReduceData<Label, Functor, ExecPolicyOrIterSpace, Reducer>;
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
