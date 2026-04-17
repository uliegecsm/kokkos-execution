#include "gtest/gtest.h"

#include "tests/utils/check_rcvr_env_queryable_with.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Tests::Utils::check_rcvr_env_queryable_with_t
 * ----------------------------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::check_rcvr_env_queryable_with and @ref Tests::Utils::check_rcvr_env_not_queryable_with.
 *
 * The tests can be found in @ref tests/utils/test_check_rcvr_env_queryable_with.cpp.
 */

namespace Tests {

constexpr struct PropA : public ::stdexec::__query<PropA> {
} prop_a{};

struct PropB : public ::stdexec::__query<PropB> { };

/**
 * @test Write @ref PropA in the environment with @c stdexec::write_env and check that @ref Tests::Utils::check_rcvr_env_queryable_with
 *       indicates that the environment is queryable with @ref PropA and
 *       @ref Tests::Utils::check_rcvr_env_not_queryable_with indicates that the environment is not queryable with @ref PropB.
 */
TEST(check_rcvr_env_queryable_with, and_not) {
    auto sndr = stdexec::just() | Tests::Utils::check_rcvr_env_queryable_with<PropA>()
              | Tests::Utils::check_rcvr_env_not_queryable_with<PropB>()
              | stdexec::write_env(stdexec::prop{prop_a, 42.});

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)
}

} // namespace Tests
