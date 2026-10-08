#include "gtest/gtest.h"

#include "kokkos-execution/impl/empty.hpp"
#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"

/**
 * @file
 *
 * Tests for @c stdexec::__closure
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::__closure.
 *
 * The tests can be found in @ref tests/stdexec/test_closure.cpp.
 */

namespace Tests {

using Empty = Kokkos::Execution::Impl::Empty;
using ThrowsWhenCopied = Tests::Utils::Functors::ThrowsWhenCopied;

//! @test The @c rvalue is properly move-constructed and can be in a constant expression.
consteval bool test_closure_from_rvalue() {
    ThrowsWhenCopied value;

    [[maybe_unused]]
    const auto res = stdexec::__closure(std::move(value));

    return true;
}

static_assert(test_closure_from_rvalue());

#if defined(__cpp_constexpr_exceptions)
//! @test It will copy-construct @c lvalues.
consteval bool test_closure_from_lvalue() {
    ThrowsWhenCopied value;
    try {
        //! This is supposed to throw.
        [[maybe_unused]]
        const auto res = stdexec::__closure(value);
        return false;
    } catch (...) {
        return true;
    }
}

static_assert(test_closure_copies_lvalues());
#endif

/**
 * @test The nothrow contract is such that a throwing-copy argument makes the closure ctor potentially-throwing,
 *       while a nothrow-movable one keeps it @c noexcept.
 */
consteval bool test_nothrow_constructibility() {
    static_assert(!std::is_nothrow_constructible_v<stdexec::__closure<ThrowsWhenCopied>, ThrowsWhenCopied&>);
    static_assert(std::is_nothrow_constructible_v<stdexec::__closure<ThrowsWhenCopied>, ThrowsWhenCopied>);
    return true;
}

static_assert(test_nothrow_constructibility());

//! @test @c stdexec::__closure decays the input types.
consteval bool test_decays() {
    static_assert(std::same_as<decltype(stdexec::__closure(std::declval<Empty&>())), stdexec::__closure<Empty>>);
    static_assert(std::same_as<decltype(stdexec::__closure(std::declval<Empty&&>())), stdexec::__closure<Empty>>);
    static_assert(std::same_as<decltype(stdexec::__closure(std::declval<const Empty&>())), stdexec::__closure<Empty>>);
    static_assert(std::same_as<
                  decltype(stdexec::__closure(std::declval<int&>(), std::declval<double&&>())),
                  stdexec::__closure<int, double>
    >);

    return true;
}

static_assert(test_decays());

class CounterTest : public testing::Test {
   protected:
    using counter_t = Tests::Utils::Functors::Counter;
   public:
    void SetUp() override {
        counter_t::reset();
    }
};

//! @test Check how things are moved/copied around with @ref Tests::Utils::Functors::Counter.
TEST_F(CounterTest, stored_by_value) {
    counter_t counter{};

    [[maybe_unused]]
    const auto r1 = stdexec::__closure(counter);
    [[maybe_unused]]
    const auto r2 = stdexec::__closure(counter);

    ASSERT_EQ(counter_t::copy_constructions, 2);
    ASSERT_EQ(counter_t::copy_assignments, 0);
    ASSERT_EQ(counter_t::move_constructions, 0);
    ASSERT_EQ(counter_t::move_assignments, 0);

    [[maybe_unused]]
    const auto r3 = stdexec::__closure(std::move(counter));

    ASSERT_EQ(counter_t::copy_constructions, 2);
    ASSERT_EQ(counter_t::copy_assignments, 0);
    ASSERT_EQ(counter_t::move_constructions, 1);
    ASSERT_EQ(counter_t::move_assignments, 0);
}

} // namespace Tests
