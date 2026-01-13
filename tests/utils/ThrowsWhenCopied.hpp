#ifndef GRAPH_DISPATCHING_TEST_UTILS_THROWSWHENCOPIED_HPP
#define GRAPH_DISPATCHING_TEST_UTILS_THROWSWHENCOPIED_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

namespace tests::utils {

/**
 * The @c Google matchers expect a @c const call operator, but the @ref sndr
 * has to be moved into the @c stdexec::sync_wait (so it has to be @c mutable).
 * This is not achievable with a lambda.
 */
template <::stdexec::sender Sndr>
struct MutableMoveToSyncWait {
    mutable Sndr sndr;

    void operator()() const {
        ::stdexec::sync_wait(std::move(sndr));
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

    ThrowsWhenCopied(const ThrowsWhenCopied&) {
        throw std::runtime_error("ThrowsWhenCopied: Throwing in copy constructor!");
    }

    KOKKOS_FUNCTION
    void operator()() const {
        Kokkos::abort("ThrowsWhenCopied: This is not intended to be called!");
    }
};

} // namespace tests::utils

#endif // GRAPH_DISPATCHING_TEST_UTILS_THROWSWHENCOPIED_HPP
