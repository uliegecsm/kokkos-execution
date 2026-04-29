#include "gtest/gtest.h"

#include "kokkos-execution/graph.hpp"

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

//! @test It won't interact with other domains inheriting from the @c stdexec::default_domain.
static_assert(!Tests::Utils::check_if_common_domain_is_default<Kokkos::Execution::GraphImpl::Domain>());

//! @test It will be default-like for @c stdexec::then_t (since it does not customize it yet).
TEST(DomainTest, default_domain_like_then) {
    static_assert(Tests::Utils::check_if_default_domain_like_for<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::then_t,
                  typename DomainTest::schedule_sender_t
    >());
}

//! @test It will be default-like for @c stdexec::bulk_t (since it does not customize it).
TEST(DomainTest, default_domain_like_bulk) {
    static_assert(Tests::Utils::check_if_default_domain_like_for<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::bulk_t,
                  typename DomainTest::schedule_sender_t,
                  stdexec::parallel_policy,
                  int
    >());
}

//! @test It has no transform for a @c stdexec::then_t sender.
TEST(DomainTest, has_no_transform_sender_for_then) {
    static_assert(!Tests::Utils::check_if_domain_has_transform_sender_for<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::then_t,
                  typename DomainTest::schedule_sender_t
    >());
}

//! @test It has no transform for a @c stdexec::bulk_t sender.
TEST(DomainTest, has_no_transform_sender_for_bulk) {
    static_assert(!Tests::Utils::check_if_domain_has_transform_sender_for<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::bulk_t,
                  typename DomainTest::schedule_sender_t,
                  stdexec::parallel_policy,
                  int
    >());
}

} // namespace Tests::GraphImpl
