#include <concepts>
#include <meta>
#include <string_view>

#include "gtest/gtest.h"

/**
 * @addtogroup unittests
 *
 * C++26 reflection
 * ----------------
 *
 * This group of tests check the C++26 reflection features.
 *
 * The tests can be found in @ref tests/std/test_reflection.cpp.
 */

namespace Tests::Std {

template <typename T>
struct MyType { };

//! @test Check that @c std::meta::display_string_of works as expected. It was introduced in @cite P2996R13.
consteval bool test_display_string_of() {
    constexpr auto str_of = std::meta::display_string_of(^^MyType<int>);

    static_assert(std::same_as<decltype(str_of), const std::string_view>);

    static_assert(str_of == "Tests::Std::MyType<int>");

    return true;
}

static_assert(test_display_string_of());

//! Because we need at least one discoverable runtime test per unit.
TEST(ReflectionTest, display_string_of) {
    std::cout << std::meta::display_string_of(^^MyType<int>) << std::endl;
}

} // namespace Tests::Std
