#ifndef KOKKOS_EXECUTION_IMPL_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_IMPL_PARALLEL_FOR_HPP

#include "kokkos-execution/impl/make_policy.hpp"
#include "kokkos-execution/impl/receiver.hpp"

/**
 * @file
 *
 * @ref Kokkos::Execution::Impl::ParallelForOpState is a trivial implementation of @ref Kokkos::Execution::parallel_for_t.
 */

namespace Kokkos::Execution {
struct parallel_for_t;
} // namespace Kokkos::Execution

namespace Kokkos::Execution::Impl {

template <typename Sndr, typename Data, typename Rcvr>
struct ParallelForOpState {
    using operation_state_concept = stdexec::operation_state_tag;

    using policy_or_iter_space_t = typename std::remove_cvref_t<Data>::policy_or_iter_space_t;

    using rcvr_t = Impl::Receiver<ParallelForOpState, stdexec::env_of_t<Rcvr>>;
    using inner_op_state_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    Data data;
    Rcvr rcvr;
    inner_op_state_t inner_op_state;

    constexpr ParallelForOpState(
        Sndr&& sndr,  // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Data&& data_, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr&& rcvr_) // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        : data(std::forward<Data>(data_))
        , rcvr(std::forward<Rcvr>(rcvr_))
        , inner_op_state(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    /**
     * We cannot use @c Kokkos::parallel_for for the default implementation, because it would build
     * a @c Kokkos::RangePolicy with a default-constructed @c Kokkos::Serial instance, whose constructor
     * aborts if @c Kokkos is not initialized.
     */
    constexpr void complete(stdexec::set_value_t) noexcept {
        static_assert(
            stdexec::__is_instance_of<policy_or_iter_space_t, Kokkos::RangePolicy>
            || iteration_space<policy_or_iter_space_t>);

        using index_type = typename make_policy_type_t<
            policy_or_iter_space_t,
            Kokkos::Execution::parallel_for_t,
            Kokkos::RangePolicy<Kokkos::Serial>
        >::index_type;

        const index_type begin = Impl::begin(data.policy_or_iter_space);
        const index_type end = Impl::end(data.policy_or_iter_space);

        for (index_type index = begin; index < end; ++index) {
            data.functor(index);
        }

        stdexec::set_value(std::move(this->rcvr));
    }

    constexpr void start() & noexcept {
        stdexec::start(inner_op_state);
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_PARALLEL_FOR_HPP
