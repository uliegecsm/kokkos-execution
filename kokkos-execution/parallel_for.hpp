#ifndef KOKKOS_EXECUTION_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_PARALLEL_FOR_HPP

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/extents.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <stdexec::sender Sndr, typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents>
struct ParallelForSender;

template <typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents>
struct ParallelForData;

} // namespace Impl

//! Custom algorithm for the @c Kokkos::parallel_for construct.
struct parallel_for_t {
    template <typename Functor, typename ExecPolicy>
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
    constexpr auto operator()(std::string label, ExecPolicy&& policy, Functor&& functor) const
        noexcept(stdexec::__nothrow_decay_copyable<Functor, ExecPolicy>) {
        return stdexec::__closure(
            *this, std::move(label), std::forward<ExecPolicy>(policy), std::forward<Functor>(functor));
    }

    template <typename Functor, typename ExecPolicy>
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<ExecPolicy>>
    constexpr auto operator()(ExecPolicy&& policy, Functor&& functor) const {
        return this->operator()("", std::forward<ExecPolicy>(policy), std::forward<Functor>(functor));
    }

    template <typename Functor, std::integral T>
    constexpr auto operator()(std::string label, const T work_count, Functor&& functor) const {
        return stdexec::__closure(
            *this, std::move(label), Kokkos::dims<1, size_t>(work_count), std::forward<Functor>(functor));
    }

    template <typename Functor, std::integral T>
    constexpr auto operator()(const T work_count, Functor&& functor) const {
        return this->operator()("", work_count, std::forward<Functor>(functor));
    }

    template <stdexec::sender Sndr, typename Functor, typename ExecPolicyOrExtents>
    requires Impl::execution_policy_or_extents<std::remove_cvref_t<ExecPolicyOrExtents>>
    constexpr auto
        operator()(Sndr&& sndr, std::string label, ExecPolicyOrExtents&& policy_or_extents, Functor&& functor) const
        noexcept(stdexec::__nothrow_decay_copyable<Sndr, Functor, ExecPolicyOrExtents>) -> Impl::ParallelForSender<
            Sndr,
            std::string,
            std::remove_cvref_t<Functor>,
            std::remove_cvref_t<ExecPolicyOrExtents>
        > {
        return {
            {parallel_for_t{},
             Impl::ParallelForData{
                 std::move(label),
                 std::forward<Functor>(functor),
                 std::forward<ExecPolicyOrExtents>(policy_or_extents)},
             std::forward<Sndr>(sndr)}
        };
    }
};

namespace Impl {

template <typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents>
struct ParallelForData {
    using label_t = Label;
    using functor_t = Functor;
    using policy_or_extents_t = ExecPolicyOrExtents;

    label_t label;
    functor_t functor;
    policy_or_extents_t policy_or_extents;
};

//! Deduction guide to store by-value.
template <typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents>
ParallelForData(Label, Functor, ExecPolicyOrExtents) -> ParallelForData<Label, Functor, ExecPolicyOrExtents>;

template <stdexec::sender Sndr, typename Label, typename Functor, execution_policy_or_extents ExecPolicyOrExtents>
struct ParallelForSender
    : stdexec::__tuple<parallel_for_t, ParallelForData<Label, Functor, ExecPolicyOrExtents>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = ParallelForData<Label, Functor, ExecPolicyOrExtents>; // NOLINT(bugprone-reserved-identifier)
    using __children_t = stdexec::__mlist<Sndr>;                           // NOLINT(bugprone-reserved-identifier)
    ///@}

    using base_t = stdexec::__tuple<parallel_for_t, __data_t, Sndr>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelForSender, Sndr, stdexec::set_error_t(std::exception_ptr))

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

inline constexpr parallel_for_t parallel_for{};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_PARALLEL_FOR_HPP
