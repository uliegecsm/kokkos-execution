#ifndef KOKKOS_EXECUTION_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_PARALLEL_FOR_HPP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core_impl;
#endif

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
    execution_policy_or_iteration_space ExecPolicyOrIterSpace
>
struct ParallelForSender;

template <typename Label, typename Functor, execution_policy_or_iteration_space ExecPolicyOrIterSpace>
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

    template <typename Functor, typename IterSpace>
    requires Impl::iteration_space<std::remove_cvref_t<IterSpace>>
    constexpr auto operator()(IterSpace&& iter_space, Functor&& functor) const {
        return stdexec::__closure(*this, "", std::forward<IterSpace>(iter_space), std::forward<Functor>(functor));
    }

    template <typename Functor, std::integral T>
    constexpr auto operator()(std::string label, const T work_count, Functor&& functor) const {
        return stdexec::__closure(
            *this,
            std::move(label),
            Kokkos::range_slice<Kokkos::detail::constant_wrapper<size_t(0)>, T>{.last = work_count},
            std::forward<Functor>(functor));
    }

    template <typename Functor, std::integral T>
    constexpr auto operator()(const T work_count, Functor&& functor) const {
        return this->operator()("", work_count, std::forward<Functor>(functor));
    }

    template <stdexec::sender Sndr, typename Functor, typename ExecPolicyOrIterSpace>
    requires Impl::execution_policy_or_iteration_space<std::remove_cvref_t<ExecPolicyOrIterSpace>>
    constexpr auto
        operator()(Sndr&& sndr, std::string label, ExecPolicyOrIterSpace&& policy_or_iter_space, Functor&& functor)
            const noexcept(stdexec::__nothrow_decay_copyable<Sndr, Functor, ExecPolicyOrIterSpace>)
                -> Impl::ParallelForSender<
                    Sndr,
                    std::string,
                    std::remove_cvref_t<Functor>,
                    std::remove_cvref_t<ExecPolicyOrIterSpace>
                > {
        return {
            {parallel_for_t{},
             Impl::ParallelForData{
                 std::move(label),
                 std::forward<Functor>(functor),
                 std::forward<ExecPolicyOrIterSpace>(policy_or_iter_space)},
             std::forward<Sndr>(sndr)}
        };
    }
};

namespace Impl {

template <typename Label, typename Functor, execution_policy_or_iteration_space ExecPolicyOrIterSpace>
struct ParallelForData {
    using label_t = Label;
    using functor_t = Functor;
    using policy_or_iter_space_t = ExecPolicyOrIterSpace;

    label_t label;
    functor_t functor;
    policy_or_iter_space_t policy_or_iter_space;
};

//! Deduction guide to store by-value.
template <typename Label, typename Functor, execution_policy_or_iteration_space ExecPolicyOrIterSpace>
ParallelForData(Label, Functor, ExecPolicyOrIterSpace) -> ParallelForData<Label, Functor, ExecPolicyOrIterSpace>;

template <
    stdexec::sender Sndr,
    typename Label,
    typename Functor,
    execution_policy_or_iteration_space ExecPolicyOrIterSpace
>
struct ParallelForSender
    : stdexec::__tuple<parallel_for_t, ParallelForData<Label, Functor, ExecPolicyOrIterSpace>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = ParallelForData<Label, Functor, ExecPolicyOrIterSpace>; // NOLINT(bugprone-reserved-identifier)
    using __children_t = stdexec::__mlist<Sndr>;                             // NOLINT(bugprone-reserved-identifier)
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
