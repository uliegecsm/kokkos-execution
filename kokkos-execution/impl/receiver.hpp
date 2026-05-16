#ifndef KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
#define KOKKOS_EXECUTION_IMPL_RECEIVER_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Receiver for an object @ref parent_op that implements @c complete.
 *
 * When this class is instantiated, @p ParentOp may be an incomplete type, but @p ParentOpBase
 * must be complete, as required for the @c stdexec::get_env call.
 *
 * Storing both the @c ParentOp* and @c ParentOpBase* pointers allows to avoid a static cast
 * from @c ParentOp* to @c ParentOpBase* or a qualified call in the @c get_env implementation,
 * which would require @p ParentOp completeness.
 */
template <typename ParentOp, typename ParentOpBase = typename ParentOp::base_t>
struct Receiver {
    using receiver_concept = SubmittedReceiverTag;

    using parent_op_base_t = ParentOpBase;

    ParentOp* parent_op;
    ParentOpBase* parent_op_base;

    constexpr explicit Receiver(ParentOp* parent_op_) noexcept
        : parent_op(parent_op_)
        , parent_op_base(parent_op_) {
    }

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
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<parent_op_base_t>> {
        return stdexec::__fwd_env(stdexec::get_env(*parent_op_base));
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_RECEIVER_HPP
