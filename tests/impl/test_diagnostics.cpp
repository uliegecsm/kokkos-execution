#include "gtest/gtest.h"

#include "kokkos-execution/impl/diagnostics.hpp"

/**
 * @addtogroup unittests
 *
 * Diagnostic utilities
 * --------------------
 *
 * This group of tests check the behavior of diagnostic utilities.
 *
 * The tests can be found in @ref tests/impl/test_diagnostics.cpp.
 */

namespace Tests::Impl {

template <typename T, bool HasDiagnose = true>
struct SomeTrait {
    static constexpr bool value = std::integral<T>;

    static constexpr auto diagnose() noexcept requires HasDiagnose
    {
        if constexpr (value) {
            return stdexec::__msuccess{};
        } else {
            return stdexec::__mexception<T>{};
        }
    }
};

using trait_ok_t = SomeTrait<int>;
using trait_ko_A_t = SomeTrait<std::string>;
using trait_ko_B_t = SomeTrait<std::string_view>;

using diagnose_ok_t = Kokkos::Execution::Impl::diagnose_t<trait_ok_t>;
using diagnose_ko_A_t = Kokkos::Execution::Impl::diagnose_t<trait_ko_A_t>;
using diagnose_ko_B_t = Kokkos::Execution::Impl::diagnose_t<trait_ko_B_t>;
using diagnose_ok_ok_t = Kokkos::Execution::Impl::diagnose_t<trait_ok_t, trait_ok_t>;
using diagnose_ok_ko_A_t = Kokkos::Execution::Impl::diagnose_t<trait_ok_t, trait_ko_A_t>;
using diagnose_ko_A_ko_B_t = Kokkos::Execution::Impl::diagnose_t<trait_ko_A_t, trait_ko_B_t>;

//! @test Check that @ref Kokkos::Execution::Impl::diagnose_t works as expected when given a single type.
consteval bool test_diagnose_t_single() {
    static_assert(std::same_as<diagnose_ok_t, stdexec::__msuccess>);
    static_assert(std::same_as<diagnose_ko_A_t, stdexec::__mexception<std::string>>);
    static_assert(std::same_as<diagnose_ko_B_t, stdexec::__mexception<std::string_view>>);

    static_assert(!stdexec::__merror<diagnose_ok_t>);
    static_assert(stdexec::__merror<diagnose_ko_A_t>);
    static_assert(stdexec::__merror<diagnose_ko_B_t>);

    return true;
}

static_assert(test_diagnose_t_single());

//! @test Check that @ref Kokkos::Execution::Impl::diagnose_t works as expected when given many types.
consteval bool test_diagnose_t_many() {
    static_assert(std::same_as<diagnose_ok_ok_t, stdexec::__mlist<stdexec::__msuccess, stdexec::__msuccess>>);
    static_assert(
        std::same_as<diagnose_ok_ko_A_t, stdexec::__mlist<stdexec::__msuccess, stdexec::__mexception<std::string>>>);
    static_assert(std::same_as<
                  diagnose_ko_A_ko_B_t,
                  stdexec::__mlist<stdexec::__mexception<std::string>, stdexec::__mexception<std::string_view>>
    >);

    //! @warning A @c stdexec::__mlist is not an error, even when it contains errors.
    static_assert(!stdexec::__merror<diagnose_ok_ok_t>);
    static_assert(!stdexec::__merror<diagnose_ok_ko_A_t>);
    static_assert(!stdexec::__merror<diagnose_ko_A_ko_B_t>);

    return true;
}

static_assert(test_diagnose_t_many());

//! @test Check that @ref Kokkos::Execution::Impl::first_error_t works as expected.
consteval bool test_first_error_t() {
    static_assert(std::same_as<Kokkos::Execution::Impl::first_error_t<diagnose_ok_t>, stdexec::__msuccess>);
    static_assert(
        std::same_as<Kokkos::Execution::Impl::first_error_t<diagnose_ko_A_t>, stdexec::__mexception<std::string>>);
    static_assert(
        std::same_as<Kokkos::Execution::Impl::first_error_t<diagnose_ko_B_t>, stdexec::__mexception<std::string_view>>);
    static_assert(std::same_as<Kokkos::Execution::Impl::first_error_t<diagnose_ok_ok_t>, stdexec::__msuccess>);
    static_assert(
        std::same_as<Kokkos::Execution::Impl::first_error_t<diagnose_ok_ko_A_t>, stdexec::__mexception<std::string>>);
    static_assert(
        std::same_as<Kokkos::Execution::Impl::first_error_t<diagnose_ko_A_ko_B_t>, stdexec::__mexception<std::string>>);

    return true;
}

static_assert(test_first_error_t());

//! @test Check that @ref Kokkos::Execution::Impl::keep_errors_t works as expected.
consteval bool test_keep_errors_t() {
    static_assert(std::same_as<Kokkos::Execution::Impl::keep_errors_t<diagnose_ok_t>, stdexec::__msuccess>);
    static_assert(
        std::same_as<Kokkos::Execution::Impl::keep_errors_t<diagnose_ko_A_t>, stdexec::__mexception<std::string>>);
    static_assert(std::same_as<Kokkos::Execution::Impl::keep_errors_t<diagnose_ok_ok_t>, stdexec::__mlist<>>);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::keep_errors_t<diagnose_ok_ko_A_t>,
                  stdexec::__mlist<stdexec::__mexception<std::string>>
    >);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::keep_errors_t<diagnose_ko_A_ko_B_t>,
                  stdexec::__mlist<stdexec::__mexception<std::string>, stdexec::__mexception<std::string_view>>
    >);

    return true;
}

static_assert(test_keep_errors_t());

template <typename T>
struct SomeTraitWithComplexDiagnose : SomeTrait<T, false> {
    static constexpr auto diagnose() noexcept {
        static_assert(false, "This should not happen.");
        return stdexec::__mexception<>{};
    }
};

/**
 * @test Check that @ref Kokkos::Execution::Impl::mandate_or_explain works as expected.
 *
 * @note We cannot check the cases where it fires, since it would generate a compile error.
 */
consteval bool test_mandate_or_explain() {
    //! When @c T::value is @c true, it succeeds.
    static_assert(Kokkos::Execution::Impl::mandate_or_explain<std::true_type>());
    static_assert(Kokkos::Execution::Impl::mandate_or_explain<SomeTrait<int>>());

    //! When @c T::value is @c false but @c T has no diagnose, it does not fail on the @c static_assert and returns @c false.
    static_assert(!Kokkos::Execution::Impl::mandate_or_explain<SomeTrait<std::string, false>>());

    //! When @c T::value is @c true, the @c T::diagnose is not instantiated, see also https://eel.is/c++draft/stmt.if#2.
    static_assert(Kokkos::Execution::Impl::mandate_or_explain<SomeTraitWithComplexDiagnose<int>>());

    return true;
}

static_assert(test_mandate_or_explain());

//! @test Check that @ref Kokkos::Execution::Impl::rebind_t works as expected.
consteval bool test_rebind_t() {
    using error_t = stdexec::__mexception<char>;
    using children_t = stdexec::__mlist<stdexec::__msuccess, error_t>;
    using rebind_t = Kokkos::Execution::Impl::rebind_t<Kokkos::Execution::Impl::WITH_CHILDREN_ERRORS, children_t>;

    static_assert(std::same_as<rebind_t, Kokkos::Execution::Impl::WITH_CHILDREN_ERRORS<stdexec::__msuccess, error_t>>);

    return true;
}

static_assert(test_rebind_t());

//! @test Check that @ref Kokkos::Execution::Impl::collapse_t works as expected.
consteval bool test_collapse_t() {
    static_assert(std::same_as<Kokkos::Execution::Impl::collapse_t<char>, char>);
    static_assert(std::same_as<Kokkos::Execution::Impl::collapse_t<stdexec::__mlist<char>>, char>);
    static_assert(
        std::same_as<Kokkos::Execution::Impl::collapse_t<stdexec::__mlist<char, char>>, stdexec::__mlist<char, char>>);

    return true;
}

static_assert(test_collapse_t());

//! Because we need at least one discoverage runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests::Impl
