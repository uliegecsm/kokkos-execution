#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_THROWS_WHEN_COPIED_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_THROWS_WHEN_COPIED_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Tests::Utils::Functors {

/**
 * The @c Google matchers expect a @c const call operator, but the @ref sndr
 * has to be moved into the @c stdexec::sync_wait (so it has to be @c mutable).
 * This is not achievable with a lambda.
 */
template <stdexec::sender Sndr>
struct MutableMoveToSyncWait {
    mutable Sndr sndr;

    void operator()() const {
        stdexec::sync_wait(std::move(sndr));
    }
};

/**
 * @brief This helper struct throws when copy constructed.
 *
 * It can be used to throw while passing it to @c Kokkos, since it always copies as seen *e.g.* in
 * https://github.com/kokkos/kokkos/blob/1e75c539491b8ce46c4671ce2e2275e15f1c27bc/core/src/Kokkos_Parallel.hpp#L142-L144
 */
struct ThrowsWhenCopied {
    ThrowsWhenCopied() = default;
    ThrowsWhenCopied& operator=(const ThrowsWhenCopied&) = default;
    ThrowsWhenCopied(ThrowsWhenCopied&&) = default;
    ThrowsWhenCopied& operator=(ThrowsWhenCopied&&) = default;
    ~ThrowsWhenCopied() = default;

    ThrowsWhenCopied(const ThrowsWhenCopied&) {
        throw std::runtime_error("ThrowsWhenCopied: Throwing in copy constructor!");
    }

    KOKKOS_FUNCTION
    void operator()() const {
        Kokkos::abort("ThrowsWhenCopied: This is not intended to be called!");
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_THROWS_WHEN_COPIED_HPP
