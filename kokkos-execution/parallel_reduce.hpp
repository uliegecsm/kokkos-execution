#ifndef KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/extents.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <
    stdexec::sender Sndr,
    typename Label,
    typename Functor,
    execution_policy_or_extents ExecPolicyOrExtents,
    typename Reducer
>
struct ParallelReduceSender;

template <typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents, typename Reducer>
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
            Kokkos::dims<1, size_t>(work_count),
            std::forward<Functor>(functor),
            std::forward<Reducer>(reducer));
    }

    template <typename Functor, std::integral T, typename Reducer>
    constexpr auto operator()(const T work_count, Functor&& functor, Reducer&& reducer) const {
        return this->operator()("", work_count, std::forward<Functor>(functor), std::forward<Reducer>(reducer));
    }

    template <stdexec::sender Sndr, typename Functor, typename ExecPolicyOrExtents, typename Reducer>
    requires Impl::execution_policy_or_extents<std::remove_cvref_t<ExecPolicyOrExtents>>
    constexpr auto operator()(
        Sndr&& sndr,
        std::string label,
        ExecPolicyOrExtents&& policy_or_extents,
        Functor&& functor,
        Reducer&& reducer) const
        noexcept(stdexec::__nothrow_decay_copyable<Sndr, Functor, ExecPolicyOrExtents, Reducer>)
            -> Impl::ParallelReduceSender<
                Sndr,
                std::string,
                std::remove_cvref_t<Functor>,
                std::remove_cvref_t<ExecPolicyOrExtents>,
                std::remove_cvref_t<Reducer>
            > {
        return {
            {parallel_reduce_t{},
             Impl::ParallelReduceData{
                 std::move(label),
                 std::forward<Functor>(functor),
                 std::forward<ExecPolicyOrExtents>(policy_or_extents),
                 std::forward<Reducer>(reducer)},
             std::forward<Sndr>(sndr)}
        };
    }
};

namespace Impl {

template <typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents, typename Reducer>
struct ParallelReduceData {
    using label_t = Label;
    using functor_t = Functor;
    using policy_or_extents_t = ExecPolicyOrExtents;
    using reducer_t = Reducer;

    static constexpr bool has_execution_policy = Kokkos::ExecutionPolicy<policy_or_extents_t>;
    static constexpr bool has_extents = Impl::extents<policy_or_extents_t>;

    template <Kokkos::ExecutionSpace Exec>
    auto policy(const Exec& exec) const {
        if constexpr (has_execution_policy) {
            static_assert(
                std::same_as<Exec, typename policy_or_extents_t::execution_space>,
                "Only the execution space instance, not its type, can be bound late.");
            return policy_or_extents_t(Kokkos::Impl::PolicyUpdate{}, policy_or_extents, exec);
        } else {
            static_assert(policy_or_extents_t::rank() == 1);
            return Kokkos::RangePolicy(exec, 0, policy_or_extents.extent(0));
        }
    }

    label_t label;
    functor_t functor;
    policy_or_extents_t policy_or_extents;
    reducer_t reducer;
};

//! Deduction guide to store by-value.
template <typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents, typename Reducer>
ParallelReduceData(Label, Functor, ExecPolicyOrExtents, Reducer)
    -> ParallelReduceData<Label, Functor, ExecPolicyOrExtents, Reducer>;

template <
    stdexec::sender Sndr,
    typename Label,
    typename Functor,
    execution_policy_or_extents ExecPolicyOrExtents,
    typename Reducer
>
struct ParallelReduceSender
    : stdexec::__tuple<parallel_reduce_t, ParallelReduceData<Label, Functor, ExecPolicyOrExtents, Reducer>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = // NOLINT(bugprone-reserved-identifier)
        ParallelReduceData<Label, Functor, ExecPolicyOrExtents, Reducer>;
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
