#ifndef KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
#define KOKKOS_EXECUTION_IMPL_RECEIVER_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

//! Receiver for an object @ref parent_op that implements @c propagate_completion_signal.
template <typename ParentOp>
struct Receiver {
    using receiver_concept = stdexec::receiver_tag;

    ParentOp* parent_op;

    void set_value() && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_error, std::forward<Error>(error));
    }

    void set_stopped() && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_stopped);
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(typename ParentOp::receiver_t, parent_op->rcvr)
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
