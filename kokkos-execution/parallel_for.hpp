#ifndef KOKKOS_EXECUTION_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_PARALLEL_FOR_HPP

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForSender;

} // namespace Impl

//! Custom algorithm for the @c Kokkos::parallel_for construct.
struct parallel_for_t {
    template <typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
    constexpr auto operator()(std::string label, ExecPolicy policy, Functor functor) const {
        return stdexec::__closure(*this, std::move(label), std::move(policy), std::move(functor));
    }

    template <typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
    constexpr auto operator()(ExecPolicy policy, Functor functor) const {
        return this->operator()("", std::move(policy), std::move(functor));
    }

    template <typename Functor, std::integral T>
    constexpr auto operator()(std::string label, const T work_count, Functor functor) const {
        using execution_space =
            typename Kokkos::Impl::FunctorPolicyExecutionSpace<std::remove_cvref_t<Functor>, void>::execution_space;
        using policy_t = Kokkos::RangePolicy<execution_space>;

        return this->operator()(std::move(label), policy_t(0, work_count), std::move(functor));
    }

    //! @warning May default to @c Kokkos::DefaultExecutionSpace, see https://github.com/kokkos/kokkos/blob/be33a115413f5eef8f82ff0ad1ca85c331edf153/core/src/Kokkos_Parallel.hpp#L155-L157.
    template <typename Functor, std::integral T>
    constexpr auto operator()(const T work_count, Functor functor) const {
        return this->operator()("", work_count, std::move(functor));
    }

    template <stdexec::sender Sndr, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
    constexpr auto operator()(Sndr&& sndr, std::string label, ExecPolicy policy, Functor functor) const {
        return Impl::ParallelForSender<Sndr, std::string, Functor, ExecPolicy>(
            {std::move(label), std::move(functor), std::move(policy)}, std::forward<Sndr>(sndr));
    }
};

namespace Impl {

template <typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForData {
    using label_t = Label;
    using functor_t = Functor;
    using policy_t = ExecPolicy;

    label_t label;
    functor_t functor;
    policy_t policy;
};

template <stdexec::sender Sndr, typename Label, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForSender : stdexec::__tuple<parallel_for_t, ParallelForData<Label, Functor, ExecPolicy>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    using base_t = stdexec::__tuple<parallel_for_t, ParallelForData<Label, Functor, ExecPolicy>, Sndr>;

    ParallelForSender(
        ParallelForData<Label, Functor, ExecPolicy> data,
        Sndr&& sndr) // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        : base_t{parallel_for_t{}, std::move(data), std::forward<Sndr>(sndr)} {
    }

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ParallelForSender, stdexec::set_error_t(std::exception_ptr))

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
