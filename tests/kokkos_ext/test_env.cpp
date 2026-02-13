#include "gtest/gtest.h"

#include "kokkos_ext/impl/env.hpp"

/**
 * @addtogroup unittests
 *
 * Environment utilities
 * ---------------------
 *
 * This group of tests check the behavior of utilities defined in @ref kokkos_ext/impl/env.hpp.
 *
 * The tests can be found in @ref tests/kokkos_ext/test_env.cpp.
 */

namespace tests::kokkos_ext {

constexpr struct PropA : public ::stdexec::__query<PropA> {
} prop_a{};

constexpr struct PropB : public ::stdexec::__query<PropB> {
} prop_b{};

constexpr struct FwdPropC
    : public ::stdexec::__query<FwdPropC>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<FwdPropC>::operator();
} fwd_prop_c{};

constexpr struct FwdPropD
    : public ::stdexec::__query<FwdPropD>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<FwdPropD>::operator();
} fwd_prop_d{};

//! @test Sanity check that only forwarding queries do pass through @c stdexec::__fwd_env, yet the resulting type also contains the unforwarded queries.
TEST(forwarding, queryable) {
    static_assert([]() {
        auto env = ::stdexec::env{
            ::stdexec::prop{    prop_a, 123},
            ::stdexec::prop{fwd_prop_c, 123}
        };

        static_assert(stdexec::__queryable_with<decltype(env), PropA>);
        static_assert(stdexec::__queryable_with<decltype(env), FwdPropC>);

        auto fwd_env = ::stdexec::__fwd_env(::stdexec::__fwd_env(env));

        static_assert(std::same_as<
                      decltype(fwd_env),
                      ::stdexec::__env::__fwd<::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::PropA, int>,
                          ::stdexec::prop<tests::kokkos_ext::FwdPropC, int>
                      >&>
        >);

        static_assert(!stdexec::__queryable_with<decltype(fwd_env), PropA>);
        static_assert(stdexec::__queryable_with<decltype(fwd_env), FwdPropC>);

        return true;
    }());
}

//! @test Set the value of a @c stdexec::prop in an empty @c stdexec::env.
TEST(upsert, set_prop_in_empty_env) {
    constexpr auto new_env = ::exec::upsert_in_env(prop_a, ::stdexec::env<>{}, 42);

    static_assert(std::same_as<decltype(new_env), const ::stdexec::env<::stdexec::prop<PropA, int>, ::stdexec::env<>>>);

    constexpr auto value = prop_a(new_env);

    static_assert(value == 42);
}

//! @test Replace the value of a single @c stdexec::prop.
TEST(upsert, replace_prop_in_single_prop) {
    static_assert([]() {
        auto env = ::stdexec::prop{prop_a, 123};

        auto new_env = ::exec::upsert_in_env(prop_a, env, 42.);

        auto value = prop_a(new_env);

        static_assert(std::same_as<decltype(new_env), ::stdexec::prop<tests::kokkos_ext::PropA, double>>);

        return value == 42.;
    }());
}

//! @test Replace the value of a @c stdexec::prop in a @c stdexec::env containing a single @c stdexec::prop.
TEST(upsert, replace_prop_in_env_with_single_prop) {
    static_assert([]() {
        auto env = ::stdexec::env{
            ::stdexec::prop{prop_a, 123},
        };

        auto new_env = ::exec::upsert_in_env(prop_a, env, 42.);

        auto value = prop_a(new_env);

        static_assert(
            std::same_as<decltype(new_env), ::stdexec::env<::stdexec::prop<tests::kokkos_ext::PropA, double>>>);

        return value == 42.;
    }());
}

//! @test Add the value of a @c stdexec::prop to another @c stdexec::prop.
TEST(upsert, add_prop_to_another_prop) {
    static_assert([]() {
        auto env = ::stdexec::prop{prop_b, 123};

        auto new_env = ::exec::upsert_in_env(prop_a, env, 42.);

        static_assert(std::same_as<
                      decltype(new_env),
                      ::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::PropA, double>,
                          ::stdexec::prop<tests::kokkos_ext::PropB, int>
                      >
        >);

        auto value = prop_a(new_env);

        return value == 42.;
    }());
}

//! @test Replace the value of a @c stdexec::prop in a @c stdexec::env containing two @c stdexec::prop.
TEST(upsert, replace_prop_in_env_with_two_props) {
    static_assert([]() {
        auto env = ::stdexec::env{
            ::stdexec::prop{prop_b, 123},
            ::stdexec::prop{prop_a, 123},
        };

        auto new_env = ::exec::upsert_in_env(prop_a, env, 42.);

        static_assert(std::same_as<
                      decltype(new_env),
                      ::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::PropB, int>,
                          ::stdexec::prop<tests::kokkos_ext::PropA, double>
                      >
        >);

        auto value = prop_a(new_env);

        return value == 42.;
    }());
}

//! @test Replace the value of a @c stdexec::prop in a forwarding @c stdexec::env containing two @c stdexec::prop.
TEST(upsert, replace_prop_in_forwarding_env_with_two_props) {
    static_assert([]() {
        auto env = ::stdexec::__fwd_env(
            ::stdexec::env{
                ::stdexec::prop{fwd_prop_d, 123},
                ::stdexec::prop{fwd_prop_c, 123},
        });

        auto new_env = ::exec::upsert_in_env(fwd_prop_c, env, 42.);

        static_assert(std::same_as<
                      decltype(new_env),
                      ::stdexec::__env::__fwd<::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::FwdPropD, int>,
                          ::stdexec::prop<tests::kokkos_ext::FwdPropC, double>
                      >>
        >);

        auto value = fwd_prop_c(new_env);

        return value == 42.;
    }());
}

/**
 * @test Add a new @c stdexec::prop to a @c stdexec::env containing multiple @c stdexec::prop.
 *
 * The resulting new environment must be queryable for this added new @c stdexec::prop.
 */
TEST(upsert, add_prop_to_env_with_multiple_props) {
    static_assert([]() {
        auto env = ::stdexec::env{
            ::stdexec::prop{    prop_b, 123},
            ::stdexec::prop{fwd_prop_c, 456},
        };

        auto new_env = ::exec::upsert_in_env(prop_a, env, 42.);

        static_assert(std::same_as<
                      decltype(new_env),
                      ::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::PropA, double>,
                          ::stdexec::env<
                              ::stdexec::prop<tests::kokkos_ext::PropB, int>,
                              ::stdexec::prop<tests::kokkos_ext::FwdPropC, int>
                          >
                      >
        >);

        return prop_a(new_env) == 42.;
    }());
}

//! @test Add a forwarding @c stdexec::prop to a forwarding @c stdexec::env.
TEST(upsert, add_fwd_prop_to_forwarding_env) {
    static_assert([]() {
        auto env = ::stdexec::__fwd_env(
            ::stdexec::env{
                ::stdexec::prop{fwd_prop_d, 123},
        });

        auto new_env = ::exec::upsert_in_env(fwd_prop_c, env, 42.);

        static_assert(std::same_as<
                      decltype(new_env),
                      ::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::FwdPropC, double>,
                          ::stdexec::__env::__fwd<stdexec::env<stdexec::prop<tests::kokkos_ext::FwdPropD, int>>>
                      >
        >);

        return fwd_prop_c(new_env) == 42.;
    }());
}

//! @test Upsert with a move-only type.
TEST(upsert, upsert_with_move_only) {
    static_assert([]() {
        auto env = ::stdexec::prop{prop_a, 123};

        struct MoveOnly : public ::stdexec::__move_only {
            int value;
            constexpr explicit MoveOnly(int value_)
                : value(value_) {
            }
        };

        auto new_env = ::exec::upsert_in_env(prop_b, env, MoveOnly(42));

        static_assert(std::same_as<
                      decltype(new_env),
                      ::stdexec::env<
                          ::stdexec::prop<tests::kokkos_ext::PropB, MoveOnly>,
                          ::stdexec::prop<tests::kokkos_ext::PropA, int>
                      >
        >);

        return prop_b(new_env).value == 42;
    }());
}

/**
 * @test Chain multiple upserts of the same property.
 *
 * The type of the environment never changes, only the query result value is updated.
 */
TEST(upsert, chain_multiple_upserts) {
    static_assert([]() {
        auto env = ::stdexec::env{
            ::stdexec::prop{prop_a, 1.},
        };

        auto new_env = ::exec::upsert_in_env(
            prop_a, ::exec::upsert_in_env(prop_a, ::exec::upsert_in_env(prop_a, env, 2.), 3.), 4.);

        static_assert(
            std::same_as<decltype(new_env), ::stdexec::env<::stdexec::prop<tests::kokkos_ext::PropA, double>>>);

        return prop_a(new_env) == 4.;
    }());
}

} // namespace tests::kokkos_ext
