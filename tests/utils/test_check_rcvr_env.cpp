#include "gtest/gtest.h"

#include "tests/utils/check_rcvr_env.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Tests::Utils::check_rcvr_env
 * -----------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::check_rcvr_env.
 *
 * The tests can be found in @ref tests/utils/test_check_rcvr_env.cpp.
 */

namespace Tests {

constexpr struct PropA : public ::stdexec::__query<PropA> {
} prop_a{};

constexpr struct PropB : public ::stdexec::__query<PropB> {
} prop_b{};

consteval auto get_sndr() {
    using expected_env_t = stdexec::env<
        const stdexec::prop<Tests::PropA, double>&,
        stdexec::__env::__fwd<stdexec::env<
            const stdexec::__env::__fwd<stdexec::prop<Tests::PropB, double>>&,
            stdexec::__env::__fwd<stdexec::__sync_wait::__env>
        >>
    >;

    return stdexec::just() | Tests::Utils::check_rcvr_env<expected_env_t>()
         | stdexec::write_env(stdexec::prop{prop_a, 42.}) | stdexec::then([]() { })
         | stdexec::write_env(stdexec::__fwd_env(stdexec::prop{prop_b, 666.}));
}

//! @test Check @ref Tests::Utils::check_rcvr_env with an rvalue sender.
TEST(check_rcvr_env, rvalue) {
    stdexec::sync_wait(get_sndr());
}

//! @test Check @ref Tests::Utils::check_rcvr_env with an lvalue sender.
TEST(check_rcvr_env, lvalue) {
    auto sndr = get_sndr();
    stdexec::sync_wait(sndr);
}

} // namespace Tests
