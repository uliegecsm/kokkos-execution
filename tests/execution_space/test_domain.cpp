#include "gtest/gtest.h"

#include "tests/utils/domain.hpp"
#include "tests/utils/execution_space_context.hpp"

/**
 * @addtogroup unittests
 *
 * Behavior of @c Kokkos::Execution::ExecutionSpaceImpl::Domain
 * ------------------------------------------------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
 *
 * The tests can be found in @ref tests/execution_space/test_domain.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

class DomainTest : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE> { };

//! @test It won't interact with other domains inheriting from the @c stdexec::default_domain.
static_assert(!Tests::Utils::check_if_common_domain_is_default<Kokkos::Execution::ExecutionSpaceImpl::Domain>());

//! @test It will always be non-default-like for @c stdexec::then_t (since it customizes it).
TEST(DomainTest, not_default_domain_like_then) {
    static_assert(!Tests::Utils::check_if_default_domain_like_for<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::then_t,
                  typename DomainTest::schedule_sender_t
    >());
}

//! @test It will always be non-default-like for @c stdexec::bulk_t (since it customizes it).
TEST(DomainTest, not_default_domain_like_bulk) {
    static_assert(!Tests::Utils::check_if_default_domain_like_for<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::bulk_t,
                  typename DomainTest::schedule_sender_t,
                  stdexec::parallel_policy,
                  int
    >());
}

//! @test It has a transform for a @c stdexec::then_t sender.
TEST(DomainTest, has_transform_sender_for_then) {
    static_assert(Tests::Utils::check_if_domain_has_transform_sender_for<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::then_t,
                  typename DomainTest::schedule_sender_t
    >());
}

//! @test It has a transform for a @c stdexec::bulk_t sender.
TEST(DomainTest, has_transform_sender_for_bulk) {
    static_assert(Tests::Utils::check_if_domain_has_transform_sender_for<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::bulk_t,
                  typename DomainTest::schedule_sender_t,
                  stdexec::parallel_policy,
                  int
    >());
}

} // namespace Tests::ExecutionSpaceImpl
