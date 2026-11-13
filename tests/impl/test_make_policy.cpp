#include "gtest/gtest.h"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "kokkos-execution/impl/make_policy.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Kokkos::Execution::Impl::make_policy
 * -------------------------------------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::Impl::make_policy.
 *
 * The tests can be found in @ref tests/impl/test_make_policy.cpp.
 */

namespace Tests::Impl {

struct MakePolicyTest
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<TEST_EXECUTION_SPACE> { };

consteval bool test_make_policy_result_t() {
    static_assert(
        std::same_as<
            Kokkos::Execution::Impl::make_policy_result_t<Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>>,
            Kokkos::SinglePolicy<TEST_EXECUTION_SPACE>
        >);

    static_assert(std::same_as<
                  Kokkos::Execution::Impl::make_policy_result_t<
                      Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>,
                      Kokkos::dims<1>
                  >,
                  Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >);

    static_assert(std::same_as<
                  Kokkos::Execution::Impl::make_policy_result_t<
                      Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE, Kokkos::IndexType<char>>,
                      Kokkos::dims<1>
                  >,
                  Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::IndexType<char>>
    >);

    static_assert(std::same_as<
                  Kokkos::Execution::Impl::make_policy_result_t<
                      Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>,
                      Kokkos::range_slice<size_t, size_t>
                  >,
                  Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >);

    static_assert(std::same_as<
                  Kokkos::Execution::Impl::make_policy_result_t<
                      Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>,
                      Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
                  >,
                  Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >);

    //! A @c Kokkos::range_slice whose compile-time range is from 0 to 1 may be converted to a @c Kokkos::SinglePolicy.
    using compile_time_range_slice_from_0_to_1_t =
        Kokkos::range_slice<std::integral_constant<size_t, 0>, std::integral_constant<size_t, 1>>;

    static_assert(std::same_as<
                  Kokkos::Execution::Impl::make_policy_result_t<
                      Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>,
                      compile_time_range_slice_from_0_to_1_t
                  >,
                  Kokkos::SinglePolicy<TEST_EXECUTION_SPACE>
    >);

    //! A @c Kokkos::range_slice whose compile-time range is from N to N+1 may be converted to a @c Kokkos::RangePolicy with @c Kokkos::LaunchBounds<1>.
    using compile_time_range_slice_from_N_to_Np1_t =
        Kokkos::range_slice<std::integral_constant<size_t, 42>, std::integral_constant<size_t, 43>>;

    static_assert(std::same_as<
                  Kokkos::Execution::Impl::make_policy_result_t<
                      Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>,
                      compile_time_range_slice_from_N_to_Np1_t
                  >,
                  Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::LaunchBounds<1>>
    >);

    return true;
}

static_assert(test_make_policy_result_t());

consteval bool test_make_policy_noexceptness() {
    static_assert(noexcept(Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>{}()));

    return true;
}

static_assert(test_make_policy_noexceptness());

//! @test Check that @ref Kokkos::Execution::Impl::make_policy works as expected.
TEST_F(MakePolicyTest, execution_space_instance) {
    const auto single_policy = Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>{}(this->exec);
    static_assert(
        std::same_as<std::remove_const_t<decltype(single_policy)>, Kokkos::SinglePolicy<TEST_EXECUTION_SPACE>>);

    const auto policy_dims1 =
        Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>{}(this->exec, Kokkos::dims<1>{10});
    static_assert(std::same_as<std::remove_const_t<decltype(policy_dims1)>, Kokkos::RangePolicy<TEST_EXECUTION_SPACE>>);
    ASSERT_EQ(policy_dims1.space(), this->exec);
    ASSERT_EQ(policy_dims1.begin(), 0);
    ASSERT_EQ(policy_dims1.end(), 10);

    const auto policy_update = Kokkos::Execution::Impl::make_policy_t<TEST_EXECUTION_SPACE>{}(
        this->exec, Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(this->exec, 0, 10));
    ASSERT_EQ(policy_update.space(), this->exec);
}

} // namespace Tests::Impl
