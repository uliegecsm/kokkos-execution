#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Environment utilities
 * ---------------------
 *
 * This group of tests check the behavior of query utilities.
 *
 * The tests can be found in @ref tests/impl/test_env.cpp.
 */

namespace Tests::Impl {

constexpr struct PropA : public stdexec::__query<PropA> {
} prop_a{};

constexpr struct FwdPropC
    : public stdexec::__query<FwdPropC>
    , stdexec::forwarding_query_t {
    using stdexec::__query<FwdPropC>::operator();
} fwd_prop_c{};

//! @test Sanity check that only forwarding queries do pass through @c stdexec::__fwd_env, yet the resulting type also contains the unforwarded queries.
TEST(forwarding, queryable) {
    static_assert([]() {
        auto env = stdexec::env{
            stdexec::prop{    prop_a, 123},
            stdexec::prop{fwd_prop_c, 123}
        };

        static_assert(stdexec::__queryable_with<decltype(env), PropA>);
        static_assert(stdexec::__queryable_with<decltype(env), FwdPropC>);

        auto fwd_env = stdexec::__fwd_env(stdexec::__fwd_env(env));

        static_assert(
            std::same_as<
                decltype(fwd_env),
                stdexec::__env::__fwd<
                    stdexec::env<stdexec::prop<Tests::Impl::PropA, int>, stdexec::prop<Tests::Impl::FwdPropC, int>>&
                >
            >);

        static_assert(!stdexec::__queryable_with<decltype(fwd_env), PropA>);
        static_assert(stdexec::__queryable_with<decltype(fwd_env), FwdPropC>);

        return true;
    }());
}

} // namespace Tests::Impl
