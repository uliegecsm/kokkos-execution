#include "gtest/gtest.h"

#include "tests/utils/domain.hpp"
#include "tests/utils/graph_context.hpp"

/**
 * @addtogroup unittests
 *
 * Behavior of @c Kokkos::Execution::GraphImpl::Domain
 * ---------------------------------------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::GraphImpl::Domain.
 *
 * The tests can be found in @ref tests/graph/test_domain.cpp.
 */

namespace Tests::GraphImpl {

class DomainTest : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE> { };

//! @test It properly interacts with other domains inheriting from the @c stdexec::default_domain.
static_assert(Tests::Utils::check_common_domain_is_default<Kokkos::Execution::GraphImpl::Domain>());

//! @test It will be default-like for @c stdexec::then_t (since it does not customize it yet).
TEST(DomainTest, default_domain_like_then) {
    static_assert(Tests::Utils::check_if_default_domain_like_then<
                  Kokkos::Execution::GraphImpl::Domain,
                  typename DomainTest::schedule_sender_t
    >());
}

} // namespace Tests::GraphImpl
