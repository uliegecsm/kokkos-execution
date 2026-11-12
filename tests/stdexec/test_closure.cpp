#include "gtest/gtest.h"

#include "kokkos-execution/impl/empty.hpp"
#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"
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
using NoOp = Tests::Utils::Functors::NoOp<false, false, false>;
using ThrowsWhenCopied = Tests::Utils::Functors::ThrowsWhenCopied;

//! @test An @c rvalue is move-constructed and can be in a constant expression.
consteval bool test_closure_from_rvalue() {
    ThrowsWhenCopied value;

    [[maybe_unused]]
    const auto res_from_xvalue = stdexec::__closure(std::move(value));

    [[maybe_unused]]
    const auto res_from_prvalue = stdexec::__closure(ThrowsWhenCopied{});

    return true;
}

static_assert(test_closure_from_rvalue());


#if defined(__cpp_constexpr_exceptions)
//! @test It will copy-construct @c lvalues.
consteval bool test_closure_from_lvalue() {
    const ThrowsWhenCopied value;
    try {
        //! This is supposed to throw.
        [[maybe_unused]]
        const auto res = stdexec::__closure(value);
        return false;
    } catch (...) {
        return true;
    }
}

static_assert(test_closure_from_lvalue());
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

/**
 * @test It may be composed with a closure of the same type.
 *
 * @note The underlying implementation relies on the @c stdexec::__composed structure, which must hence be instantiable
 *       for closures of the same type.
 *
 * See also:
 * - https://github.com/NVIDIA/stdexec/issues/2440
 * - https://github.com/NVIDIA/stdexec/blob/a8449abd18f580b595bcacf249b5c18b9b8a5b2b/include/stdexec/__detail/__sender_adaptor_closure.hpp#L54-L57
 * - https://github.com/NVIDIA/stdexec/blob/a8449abd18f580b595bcacf249b5c18b9b8a5b2b/include/stdexec/__detail/__sender_adaptor_closure.hpp#L73
 * - https://github.com/NVIDIA/stdexec/blob/a8449abd18f580b595bcacf249b5c18b9b8a5b2b/include/stdexec/__detail/__sender_adaptor_closure.hpp#L90
 */
consteval bool test_may_be_composed_with_closure_of_the_same_type() {
    auto clsr_a = stdexec::then(NoOp{});
    auto clsr_b = stdexec::then(NoOp{});

    using clsr_t = decltype(clsr_a);
    static_assert(std::same_as<clsr_t, stdexec::__closure<stdexec::then_t, NoOp>>);

    const auto clsr_composed = clsr_a | clsr_b;

    static_assert(
        std::same_as<std::remove_const_t<decltype(clsr_composed)>, stdexec::__clsur::__composed<clsr_t, clsr_t>>);
    static_assert(std::derived_from<decltype(clsr_composed), stdexec::__composed<clsr_t, clsr_t>>);
    static_assert(stdexec::__sender_adaptor_closure<decltype(clsr_composed)>);

    return true;
}

static_assert(test_may_be_composed_with_closure_of_the_same_type());

class ClosureTest : public testing::Test {
   protected:
    using counter_t = Tests::Utils::Functors::Counter;
   public:
    void SetUp() override {
        counter_t::reset();
    }
};

//! @test Check how things are moved/copied around with @ref Tests::Utils::Functors::Counter.
TEST_F(ClosureTest, stored_by_value) {
    counter_t counter{};

    [[maybe_unused]]
    const auto res1 = stdexec::__closure(counter);
    [[maybe_unused]]
    const auto res2 = stdexec::__closure(counter);

    ASSERT_EQ(counter_t::copy_constructions, 2);
    ASSERT_EQ(counter_t::copy_assignments, 0);
    ASSERT_EQ(counter_t::move_constructions, 0);
    ASSERT_EQ(counter_t::move_assignments, 0);

    [[maybe_unused]]
    const auto res3 = stdexec::__closure(std::move(counter));

    ASSERT_EQ(counter_t::copy_constructions, 2);
    ASSERT_EQ(counter_t::copy_assignments, 0);
    ASSERT_EQ(counter_t::move_constructions, 1);
    ASSERT_EQ(counter_t::move_assignments, 0);

    [[maybe_unused]]
    const auto res4 = stdexec::__closure(counter_t{});

    ASSERT_EQ(counter_t::copy_constructions, 2);
    ASSERT_EQ(counter_t::copy_assignments, 0);
    ASSERT_EQ(counter_t::move_constructions, 2);
    ASSERT_EQ(counter_t::move_assignments, 0);
}

} // namespace Tests
