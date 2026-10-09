#ifndef KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP
#define KOKKOS_EXECUTION_PARALLEL_REDUCE_HPP

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy, typename Reducer>
struct ParallelReduceSender;

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy, typename Reducer>
struct ParallelReduceData;

} // namespace Impl

//! Custom algorithm for the @c Kokkos::parallel_reduce construct.
struct parallel_reduce_t {
    template <typename Functor, typename ExecPolicy, typename Reducer>
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
    constexpr auto operator()(std::string label, ExecPolicy&& policy, Functor&& functor, Reducer&& reducer) const {
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
        using execution_space =
            typename Kokkos::Impl::FunctorPolicyExecutionSpace<std::remove_cvref_t<Functor>, void>::execution_space;
        using policy_t = Kokkos::RangePolicy<execution_space>;

        return this->operator()(
            std::move(label), policy_t(0, work_count), std::forward<Functor>(functor), std::forward<Reducer>(reducer));
    }

    //! @warning May default to @c Kokkos::DefaultExecutionSpace, see https://github.com/kokkos/kokkos/blob/be33a115413f5eef8f82ff0ad1ca85c331edf153/core/src/Kokkos_Parallel.hpp#L155-L157.
    template <typename Functor, std::integral T, typename Reducer>
    constexpr auto operator()(const T work_count, Functor&& functor, Reducer&& reducer) const {
        return this->operator()("", work_count, std::forward<Functor>(functor), std::forward<Reducer>(reducer));
    }

    template <stdexec::sender Sndr, typename Functor, typename ExecPolicy, typename Reducer>
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
    constexpr auto
        operator()(Sndr&& sndr, std::string label, ExecPolicy&& policy, Functor&& functor, Reducer&& reducer) const
        noexcept(stdexec::__nothrow_decay_copyable<Sndr, Functor, ExecPolicy, Reducer>) -> Impl::ParallelReduceSender<
            Sndr,
            std::string,
            std::remove_cvref_t<Functor>,
            std::remove_cvref_t<ExecPolicy>,
            std::remove_cvref_t<Reducer>
        > {
        return {
            {parallel_reduce_t{},
             Impl::ParallelReduceData{
                 std::move(label),
                 std::forward<Functor>(functor),
                 std::forward<ExecPolicy>(policy),
                 std::forward<Reducer>(reducer)},
             std::forward<Sndr>(sndr)}
        };
    }
};

namespace Impl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy, typename Reducer>
struct ParallelReduceData {
    using label_t = Label;
    using functor_t = Functor;
    using policy_t = ExecPolicy;
    using reducer_t = Reducer;

    label_t label;
    functor_t functor;
    policy_t policy;
    reducer_t reducer;
};

//! Deduction guide to store by-value.
template <typename Label, typename Functor, typename ExecPolicy, typename Reducer>
ParallelReduceData(Label, Functor, ExecPolicy, Reducer) -> ParallelReduceData<Label, Functor, ExecPolicy, Reducer>;

template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy, typename Reducer>
struct ParallelReduceSender
    : stdexec::__tuple<parallel_reduce_t, ParallelReduceData<Label, Functor, ExecPolicy, Reducer>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __tag_t = parallel_for_t;                                           // NOLINT(bugprone-reserved-identifier)
    using __data_t = ParallelReduceData<Label, Functor, ExecPolicy, Reducer>; // NOLINT(bugprone-reserved-identifier)
    using __children_t = stdexec::__mlist<Sndr>;                              // NOLINT(bugprone-reserved-identifier)
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

#endif // KOKKOS_EXECUTION_PARALLEL_FOR_HPP
