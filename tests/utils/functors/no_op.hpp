#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_NO_OP_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_NO_OP_HPP

#include "Kokkos_Core.hpp"

namespace Tests::Utils::Functors {

//! Does nothing.
template <bool MayThrowOnCall = true, bool MayThrowOnCopy = true, bool MayThrowOnMove = true>
struct NoOp {
    NoOp() = default;

    NoOp(const NoOp&) noexcept(!MayThrowOnCopy) { // NOLINT(modernize-use-equals-default)
    }

    NoOp(NoOp&&) noexcept(!MayThrowOnMove) {
    }

    NoOp& operator=(const NoOp&) noexcept(!MayThrowOnCopy) { // NOLINT(modernize-use-equals-default)
        return *this;
    }

    NoOp& operator=(NoOp&&) noexcept(!MayThrowOnMove) {
        return *this;
    }

    ~NoOp() = default;

    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args&&...) const noexcept(!MayThrowOnCall) {
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_NO_OP_HPP
