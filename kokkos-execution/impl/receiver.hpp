#ifndef KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
#define KOKKOS_EXECUTION_IMPL_RECEIVER_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::Impl {

//! Receiver for an object @ref parent_op that implements @c complete.
template <typename ParentOp, typename Env = stdexec::env_of_t<ParentOp>>
struct Receiver {
    using receiver_concept = SubmittedReceiverTag;

    ParentOp* parent_op;

    void set_value() && noexcept {
        parent_op->submit();
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept {
        parent_op->complete(stdexec::set_error, std::forward<Error>(error));
    }

    void set_stopped() && noexcept {
        parent_op->complete(stdexec::set_stopped);
    }

    void submitted() && noexcept {
        parent_op->submit();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<Env> {
        return stdexec::__fwd_env(stdexec::get_env(*parent_op));
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
