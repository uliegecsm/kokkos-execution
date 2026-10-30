#include "gtest/gtest.h"

#include "kokkos-execution/impl/label.hpp"
#include "kokkos-execution/impl/properties.hpp"

#include "tests/utils/properties.hpp"

/**
 * @addtogroup unittests
 *
 * Properties utilities
 * --------------------
 *
 * This group of tests check utilities from @ref kokkos-execution/impl/properties.hpp.
 *
 * The tests can be found in @ref tests/impl/test_properties.cpp.
 */

namespace Tests::Impl {

//! @test Check that @ref Kokkos::Execution::Impl::has_property works as expected.
consteval bool test_has_property() {
    static_assert(Kokkos::Execution::Impl::has_property<
                  Tests::Utils::LabeledData<std::string>,
                  Kokkos::Execution::Impl::Label<std::string>
    >());
    static_assert(
        Kokkos::Execution::Impl::has_property<Tests::Utils::LabeledData<std::string>, Kokkos::Execution::Impl::Label>());
    static_assert(!Kokkos::Execution::Impl::has_property<Tests::Utils::LabeledData<std::string>, Kokkos::ChunkSize>());

    static_assert(
        !Kokkos::Execution::Impl::has_property<Tests::Utils::MyPropA, Kokkos::Execution::Impl::Label<std::string>>());
    static_assert(!Kokkos::Execution::Impl::has_property<
                  Tests::Utils::MyPropA,
                  Kokkos::Execution::Impl::Label<std::string_view>
    >());

    static_assert(Kokkos::Execution::Impl::has_property<
                  Tests::Utils::MultiPropData<std::string>,
                  Kokkos::Execution::Impl::Label<std::string>
    >());
    static_assert(Kokkos::Execution::Impl::has_property<
                  Tests::Utils::MultiPropData<std::string_view>,
                  Kokkos::Execution::Impl::Label<std::string_view>
    >());
    static_assert(
        Kokkos::Execution::Impl::has_property<Tests::Utils::MultiPropData<std::string>, Tests::Utils::MyPropA>());
    static_assert(
        Kokkos::Execution::Impl::has_property<Tests::Utils::MultiPropData<std::string>, Tests::Utils::MyPropB>());

    return true;
}

static_assert(test_has_property());

//! @test Check that @ref Kokkos::Execution::Impl::get_property works as expected.
consteval bool test_get_property() {
    using labeled_data_t = Tests::Utils::LabeledData<std::string_view>;
    using multi_data_t = Tests::Utils::MultiPropData<std::string_view>;

    // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TEST_GET_PROPERTY(_prop_, _data_t_, _value_t_)                                                                 \
    static_assert(                                                                                                     \
        std::same_as<decltype(Kokkos::Execution::Impl::get_property<_prop_>(std::declval<_data_t_>())), _value_t_>);

    //! A non-template type property.
    TEST_GET_PROPERTY(Tests::Utils::MyPropA, const multi_data_t&, const std::string&)
    TEST_GET_PROPERTY(Tests::Utils::MyPropA, multi_data_t&, std::string&)
    TEST_GET_PROPERTY(Tests::Utils::MyPropA, multi_data_t&&, std::string&&)

    //! A template type property.
    TEST_GET_PROPERTY(Kokkos::Execution::Impl::Label, const labeled_data_t&, const std::string_view&)
    TEST_GET_PROPERTY(Kokkos::Execution::Impl::Label, labeled_data_t&, std::string_view&)
    TEST_GET_PROPERTY(Kokkos::Execution::Impl::Label, labeled_data_t&&, std::string_view&&)

#undef TEST_GET_PROPERTY

    constexpr labeled_data_t data{"my-nice-label"};
    constexpr auto value = Kokkos::Execution::Impl::get_property<Kokkos::Execution::Impl::Label>(data);
    static_assert(value == "my-nice-label");

    return true;
}

static_assert(test_get_property());

/**
 * @test Check that @ref Kokkos::Execution::Impl::get_property can be used to
 *       retrieve a @ref Kokkos::Execution::Impl::Label (at runtime).
 */
TEST(get_property, label) {
    ASSERT_EQ(
        Kokkos::Execution::Impl::get_property<Kokkos::Execution::Impl::Label>(
            Tests::Utils::LabeledData<std::string>{"my-nice-label"}),
        "my-nice-label");
}

} // namespace Tests::Impl
