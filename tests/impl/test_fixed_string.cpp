#include "gtest/gtest.h"

#include <string>

#include "kokkos-execution/impl/fixed_string.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Kokkos::Execution::Impl::FixedString
 * -------------------------------------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::Impl::FixedString.
 *
 * The tests can be found in @ref tests/impl/test_fixed_string.cpp
 */

namespace Tests::Impl {

using Kokkos::Execution::Impl::FixedString;
using namespace std::string_view_literals;

//! @test Check that @ref Kokkos::Execution::Impl::is_fixed_string_v works as expected.
consteval bool test_FixedString_is_fixed_string() {
    static_assert(Kokkos::Execution::Impl::is_fixed_string_v<FixedString<42>>);
    static_assert(!Kokkos::Execution::Impl::is_fixed_string_v<char>);

    return true;
}

static_assert(test_FixedString_is_fixed_string());

//! @test Check that @ref Kokkos::Execution::Impl::FixedString properly deduces its capacity from the input literal.
consteval bool test_FixedString_deduces_capacity_from_literal() {
    static_assert(std::same_as<decltype(FixedString{""}), FixedString<1>>);
    static_assert(std::same_as<decltype(FixedString{"a"}), FixedString<2>>);
    static_assert(std::same_as<decltype(FixedString{"ab"}), FixedString<3>>);
    static_assert(std::same_as<decltype(FixedString{"abc"}), FixedString<4>>);

    return true;
}

static_assert(test_FixedString_deduces_capacity_from_literal());

//! @test Check the layout of @ref Kokkos::Execution::Impl::FixedString.
template <size_t Size>
consteval bool test_FixedString_layout() {
    static_assert(sizeof(FixedString<Size>) == Size);
    static_assert(std::is_trivially_copyable_v<FixedString<Size>>);
    static_assert(std::is_standard_layout_v<FixedString<Size>>);

    return true;
}

static_assert(test_FixedString_layout<1>());
static_assert(test_FixedString_layout<7>());

//! @test Check how @ref Kokkos::Execution::Impl::FixedString can be constructed.
consteval bool test_FixedString_is_constructible_from_matching_char_array_only() {
    static_assert(std::is_constructible_v<FixedString<4>, const char (&)[4]>);

    //! It cannot be constructed from a mismatch-in-size array.
    static_assert(!std::is_constructible_v<FixedString<4>, const char (&)[5]>);

    //! It cannot be constructed from a @c char pointer.
    static_assert(!std::is_constructible_v<FixedString<4>, const char*>);

    static_assert(!std::is_constructible_v<FixedString<4>, std::string_view>);
    static_assert(!std::is_constructible_v<FixedString<4>, std::string>);

    return true;
}

static_assert(test_FixedString_is_constructible_from_matching_char_array_only());

//! @test Check @ref Kokkos::Execution::Impl::FixedString constructed from a literal.
consteval bool test_FixedString_from_literal() {
    constexpr FixedString fstr{"label"};
    static_assert(fstr.view() == "label"sv);
    static_assert(fstr.size() == 5);

    return true;
}

static_assert(test_FixedString_from_literal());

//! @test Check @ref Kokkos::Execution::Impl::FixedString constructed from an empty literal.
consteval bool test_FixedString_from_empty_literal() {
    constexpr FixedString fstr{""};
    static_assert(fstr.size() == 0);
    static_assert(fstr.view().empty());
    static_assert(fstr.begin() == fstr.end());

    return true;
}

static_assert(test_FixedString_from_empty_literal());

/**
 * @test Check whether @ref Kokkos::Execution::Impl::FixedString::view is @c noexcept or not.
 *
 * The standard does not specify that the @c std::string_view constructor from a pointer and a size is
 * @c noexcept. Yet, implementers generally mark it @c noexcept.
 */
consteval bool test_view_is_noexcept() {
    static_assert(noexcept(FixedString{"abc"}.view()));

    return true;
}

static_assert(test_view_is_noexcept());

//! @test Check that @ref Kokkos::Execution::Impl::FixedString properly converts to @c std::string and @c std::string_view
TEST(FixedStringTest, converts_to_string_view_and_string) {
    static_assert(std::is_constructible_v<std::string_view, const FixedString<6>&>);

    //! Useful *e.g.* for https://github.com/kokkos/kokkos/blob/2c577ed795cf8eb36a665d1c15d722ef6461d1b1/core/src/Kokkos_Parallel_For.hpp#L50.
    static_assert(std::is_constructible_v<std::string, const FixedString<6>&>);

    constexpr FixedString fstr{"label"};

    const std::string_view fstr_view = fstr;
    EXPECT_EQ(fstr_view, "label"sv);
    EXPECT_EQ(std::string(fstr), "label");
}

template <FixedString S>
struct Tag {
    static constexpr std::string_view value = S.view();
};

//! @test Check that @ref Kokkos::Execution::Impl::FixedString can be used as a template parameter.
consteval bool test_FixedString_is_usable_as_a_template_parameter() {
    static_assert(Tag<"abcd">::value == "abcd"sv);
    static_assert(std::same_as<Tag<"abcd">, Tag<"abcd">>);
    static_assert(!std::same_as<Tag<"abcd">, Tag<"abdc">>);

    return true;
}

static_assert(test_FixedString_is_usable_as_a_template_parameter());

#if defined(__cpp_lib_is_structural)
//! @test Check that @ref Kokkos::Execution::Impl::FixedString is a structural type.
consteval bool test_FixedString_is_structural() {
    static_assert(std::is_structural_v<FixedString{"abcd"}>);

    return true;
}

static_assert(test_FixedString_is_structural());
#endif

} // namespace Tests::Impl
