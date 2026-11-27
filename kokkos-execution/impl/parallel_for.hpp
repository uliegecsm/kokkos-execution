#ifndef KOKKOS_EXECUTION_IMPL_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_IMPL_PARALLEL_FOR_HPP

#include "kokkos-execution/impl/immovable.hpp"
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
struct ParallelForOpState : public Impl::Immovable {
    using operation_state_concept = stdexec::operation_state_tag;

    using functor_t = typename std::remove_cvref_t<Data>::functor_t;
    using policy_or_iter_space_t = typename std::remove_cvref_t<Data>::policy_or_iter_space_t;

    static_assert(
        stdexec::__is_instance_of<policy_or_iter_space_t, Kokkos::RangePolicy>
        || iteration_space<policy_or_iter_space_t>);

    using rcvr_t = Impl::Receiver<ParallelForOpState, stdexec::env_of_t<Rcvr>>;
    using inner_op_state_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    using policy_t = make_policy_type_t<
        policy_or_iter_space_t,
        Kokkos::Execution::parallel_for_t,
        Kokkos::RangePolicy<Kokkos::Serial>
    >;

    using index_t = typename policy_t::index_type;

    Data data;
    Rcvr rcvr;
    inner_op_state_t inner_op_state;

    constexpr explicit ParallelForOpState(
        Sndr&& sndr,  // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Data&& data_, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr_)
        : data(std::forward<Data>(data_))
        , rcvr(std::move(rcvr_))
        , inner_op_state(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    constexpr void __run() noexcept(stdexec::__nothrow_applicable<functor_t&, index_t>)
        requires std::is_void_v<typename policy_t::work_tag>
    {
        const index_t begin = Impl::begin(data.policy_or_iter_space);
        const index_t end = Impl::end(data.policy_or_iter_space);

        for (index_t index = begin; index < end; ++index)
            data.functor(index);
    }

    constexpr void
        __run() noexcept(stdexec::__nothrow_applicable<functor_t&, const typename policy_t::work_tag&, index_t>) {
        const index_t begin = Impl::begin(data.policy_or_iter_space);
        const index_t end = Impl::end(data.policy_or_iter_space);

        const typename policy_t::work_tag tag{};

        for (index_t index = begin; index < end; ++index)
            data.functor(tag, index);
    }

    /**
     * We cannot use @c Kokkos::parallel_for for the default implementation, because it would build
     * a @c Kokkos::RangePolicy with a default-constructed @c Kokkos::Serial instance, whose constructor
     * aborts if @c Kokkos is not initialized.
     */
    constexpr void complete(stdexec::set_value_t) noexcept {
        if constexpr (noexcept(this->__run())) {
            this->__run();
        } else {
            try {
                this->__run();
            } catch (...) {
                stdexec::set_error(std::move(this->rcvr), std::current_exception());
                return;
            }
        }

        stdexec::set_value(std::move(this->rcvr));
    }

    template <typename Error>
    constexpr void complete(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(this->rcvr), std::forward<Error>(error));
    }

    constexpr void start() & noexcept {
        stdexec::start(inner_op_state);
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_PARALLEL_FOR_HPP
