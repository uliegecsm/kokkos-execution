#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP

#include "tests/utils/stdexec.hpp"

namespace Tests::Utils {

//! @test Check whether the sender can be nothrow applied for @c stdexec::sync_wait.
template <typename Domain, stdexec::sender Sndr>
consteval bool check_nothrow_apply_sender() {
    static_assert(stdexec::__never_sends<stdexec::set_error_t, Sndr>);
    static_assert(Tests::Utils::has_nothrow_apply_sender<Domain, stdexec::sync_wait_t, Sndr>);
    static_assert(!Tests::Utils::has_nothrow_apply_sender<stdexec::sync_wait_t, Sndr>);
    return true;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP
