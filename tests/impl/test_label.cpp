#include "gtest/gtest.h"

#include "kokkos-execution/impl/label.hpp"

#include "tests/utils/properties.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Kokkos::Execution::Impl::Label
 * -------------------------------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::Impl::Label and related utilities.
 *
 * The tests can be found in @ref tests/impl/test_label.cpp.
 */

namespace Tests::Impl {

//! @test Check that @ref Kokkos::Execution::Impl::label works as expected.
consteval bool test_label() {
    static_assert(Kokkos::Execution::Impl::label<std::string>);
    static_assert(Kokkos::Execution::Impl::label<char[9]>);
    static_assert(Kokkos::Execution::Impl::label<const char[9]>);
    static_assert(Kokkos::Execution::Impl::label<const char*>);

    static_assert(!Kokkos::Execution::Impl::label<std::string_view>);
    static_assert(!Kokkos::Execution::Impl::label<int>);

    return true;
}

static_assert(test_label());

/**
 * @test Check that @ref Kokkos::Execution::Impl::Label CTAD works as expected.
 *
 * Any convertible-to @c std::string type maps to @ref Kokkos::Execution::Impl::Label<std::string>.
 */
consteval bool test_label_ctad() {
    using expt_t = Kokkos::Execution::Impl::Label<std::string>;

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::Label{"literal"}), expt_t>);

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::Label{std::declval<char[9]>()}), expt_t>);

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::Label{std::declval<const char[9]>()}), expt_t>);

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::Label{std::declval<const char*>()}), expt_t>);

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::Label{std::declval<std::string>()}), expt_t>);

    static_assert(std::same_as<
                  decltype(Kokkos::Execution::Impl::Label{std::declval<std::string_view>()}),
                  Kokkos::Execution::Impl::Label<std::string_view>
    >);

    return true;
}

static_assert(test_label_ctad());

//! @test Check that @ref Kokkos::Execution::Impl::labeled works as expected.
consteval bool test_labeled() {
    static_assert(Kokkos::Execution::Impl::labeled<Tests::Utils::LabeledData<std::string>>);
    static_assert(Kokkos::Execution::Impl::labeled<Tests::Utils::LabeledData<std::string_view>>);
    static_assert(Kokkos::Execution::Impl::labeled<Tests::Utils::MultiPropData<std::string>>);
    static_assert(Kokkos::Execution::Impl::labeled<Tests::Utils::MultiPropData<std::string_view>>);

    static_assert(!Kokkos::Execution::Impl::labeled<Tests::Utils::MyPropA>);

    return true;
}

static_assert(test_labeled());

//! @c constinit requires static initialization.
constinit const auto constinit_label = Kokkos::Execution::label("my label");

//! @test Check that @ref Kokkos::Execution::label works as expected.
consteval bool test_static_label() {
    constexpr auto my_label = Kokkos::Execution::label("my label");

    static_assert(std::same_as<decltype(my_label), const Kokkos::Execution::Impl::Label<std::string_view>>);
    static_assert(my_label.m_value == "my label");

    return true;
}

static_assert(test_static_label());

//! @test Check the value of @ref Kokkos::Execution::label at runtime.
TEST(LabelTest, static_label) {
    ASSERT_EQ(constinit_label.m_value, "my label");
}

//! A type that is implicitly convertible to a @c std::string, but is not one.
struct ConvertibleToString {
    std::string m_value;

    constexpr operator std::string() const noexcept { // NOLINT(google-explicit-constructor)
        return m_value;
    }
};

//! @test Check that @ref Kokkos::Execution::Impl::as_convertible_to_string forwards @c std::string as-is.
consteval bool test_as_convertible_to_string_std_string() {
    std::string lvalue{"my label"};
    const std::string const_lvalue{"my label"};

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string(lvalue)), std::string&>);
    static_assert(
        std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string(const_lvalue)), const std::string&>);
    static_assert(
        std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string(std::move(lvalue))), std::string&&>);

    static_assert(std::addressof(Kokkos::Execution::Impl::as_convertible_to_string(lvalue)) == std::addressof(lvalue));
    static_assert(
        std::addressof(Kokkos::Execution::Impl::as_convertible_to_string(const_lvalue))
        == std::addressof(const_lvalue));

    return true;
}

static_assert(test_as_convertible_to_string_std_string());

//! @test Check that @ref Kokkos::Execution::Impl::as_convertible_to_string forwards implicitly convertible types as-is.
consteval bool test_as_convertible_to_string_convertible() {
    const ConvertibleToString const_lvalue{"my label"};
    ConvertibleToString lvalue{"my label"};

    static_assert(
        std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string(lvalue)), ConvertibleToString&>);
    static_assert(std::same_as<
                  decltype(Kokkos::Execution::Impl::as_convertible_to_string(const_lvalue)),
                  const ConvertibleToString&
    >);
    static_assert(std::same_as<
                  decltype(Kokkos::Execution::Impl::as_convertible_to_string(std::move(lvalue))),
                  ConvertibleToString&&
    >);

    static_assert(std::addressof(Kokkos::Execution::Impl::as_convertible_to_string(lvalue)) == std::addressof(lvalue));
    static_assert(
        std::addressof(Kokkos::Execution::Impl::as_convertible_to_string(const_lvalue))
        == std::addressof(const_lvalue));

    static_assert(
        std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string("my label")), const char (&)[9]>);

    return true;
}

static_assert(test_as_convertible_to_string_convertible());

//! @test Check that @ref Kokkos::Execution::Impl::as_convertible_to_string materializes a @c std::string only when needed.
consteval bool test_as_convertible_to_string_not_convertible_will_materialize() {
    constexpr std::string_view view{"my label"};
    constexpr Kokkos::Execution::Impl::FixedString fixed{"my label"};

    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string(view)), std::string>);
    static_assert(std::same_as<decltype(Kokkos::Execution::Impl::as_convertible_to_string(fixed)), std::string>);
    static_assert(Kokkos::Execution::Impl::as_convertible_to_string(view) == "my label");
    static_assert(Kokkos::Execution::Impl::as_convertible_to_string(fixed) == "my label");
    static_assert(Kokkos::Execution::Impl::as_convertible_to_string(std::string_view{}).empty());

    return true;
}

static_assert(test_as_convertible_to_string_not_convertible_will_materialize());

} // namespace Tests::Impl
