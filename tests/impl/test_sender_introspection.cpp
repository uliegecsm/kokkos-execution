#include "gtest/gtest.h"

#include "kokkos-execution/impl/sender_introspection.hpp"

/**
 * @addtogroup unittests
 *
 * Sender introspection utilities
 * ------------------------------
 *
 * This group of tests check the behavior of sender introspection utilities
 * from @ref kokkos-execution/impl/sender_introspection.hpp.
 *
 * The tests can be found in @ref tests/impl/test_sender_introspection.cpp.
 */

namespace Tests::Impl {

struct HasChildrenTest : public testing::Test {
    using sndr_without_child_t = decltype(stdexec::just());
    using sndr_with_one_child_t = decltype(stdexec::just() | stdexec::then([]() { }));
    using sndr_with_children_t = decltype(stdexec::when_all(stdexec::just(), stdexec::just()));
};

//! @test Check @ref Kokkos::Execution::Impl::has_child and @ref Kokkos::Execution::Impl::has_children when there is zero child.
TEST_F(HasChildrenTest, zero) {
    static_assert(!Kokkos::Execution::Impl::has_child<sndr_without_child_t>);
    static_assert(!Kokkos::Execution::Impl::has_children<sndr_without_child_t>);
}

//! @test Check @ref Kokkos::Execution::Impl::has_child and @ref Kokkos::Execution::Impl::has_children when there is one child.
TEST_F(HasChildrenTest, one) {
    static_assert(Kokkos::Execution::Impl::has_child<sndr_with_one_child_t>);
    static_assert(!Kokkos::Execution::Impl::has_children<sndr_with_one_child_t>);
}

//! @test Check @ref Kokkos::Execution::Impl::has_child and @ref Kokkos::Execution::Impl::has_children when there are children.
TEST_F(HasChildrenTest, more_than_one) {
    static_assert(!Kokkos::Execution::Impl::has_child<sndr_with_children_t>);
    static_assert(Kokkos::Execution::Impl::has_children<sndr_with_children_t>);
}

struct RemainsOnTest : public HasChildrenTest { };

//! @test Check @ref Kokkos::Execution::Impl::remains_on.
TEST_F(RemainsOnTest, default_domain) {
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_without_child_t, stdexec::env<>>,
                  stdexec::default_domain
    >);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_with_one_child_t, stdexec::env<>>,
                  stdexec::default_domain
    >);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_with_children_t, stdexec::env<>>,
                  stdexec::default_domain
    >);

    static_assert(Kokkos::Execution::Impl::remains_on<
                  stdexec::set_value_t,
                  sndr_without_child_t,
                  stdexec::default_domain,
                  stdexec::env<>
    >);
    static_assert(Kokkos::Execution::Impl::remains_on<
                  stdexec::set_value_t,
                  sndr_without_child_t,
                  stdexec::indeterminate_domain<>
    >);

    static_assert(Kokkos::Execution::Impl::remains_on<
                  stdexec::set_value_t,
                  sndr_with_one_child_t,
                  stdexec::default_domain,
                  stdexec::env<>
    >);
    static_assert(Kokkos::Execution::Impl::remains_on<
                  stdexec::set_value_t,
                  sndr_with_one_child_t,
                  stdexec::indeterminate_domain<>
    >);

    static_assert(Kokkos::Execution::Impl::remains_on<
                  stdexec::set_value_t,
                  sndr_with_children_t,
                  stdexec::default_domain,
                  stdexec::env<>
    >);
    static_assert(Kokkos::Execution::Impl::remains_on<
                  stdexec::set_value_t,
                  sndr_with_children_t,
                  stdexec::indeterminate_domain<>
    >);
}

} // namespace Tests::Impl
