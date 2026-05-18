#ifndef KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
#define KOKKOS_EXECUTION_IMPL_RECEIVER_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::Impl {

//! Receiver for an object @ref parent_op that implements @c complete.
template <typename ParentOp, typename ParentOpBase = typename ParentOp::base_t>
struct Receiver {
    using receiver_concept = SubmittedReceiverTag;

    using parent_op_base_t = ParentOpBase;

    parent_op_base_t* parent_op_base;

    constexpr explicit Receiver(ParentOp* parent_op) noexcept
        : parent_op_base(parent_op) {
    }

   private:
    constexpr ParentOp* parent_op() const noexcept {
        return static_cast<ParentOp*>(parent_op_base);
    }

   public:
    void set_value() && noexcept {
        parent_op()->submit();
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept {
        parent_op()->complete(stdexec::set_error, std::forward<Error>(error));
    }

    void set_stopped() && noexcept {
        parent_op()->complete(stdexec::set_stopped);
    }

    void submitted() && noexcept {
        parent_op()->submit();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<parent_op_base_t>> {
        return stdexec::__fwd_env(stdexec::get_env(*parent_op_base));
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
