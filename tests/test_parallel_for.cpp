#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/parallel_for.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/properties.hpp"

/**
 * @addtogroup unittests
 *
 * Sender adaptor @c Kokkos::Execution::parallel_for
 * -------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::parallel_for.
 *
 * The tests can be found in @ref tests/test_parallel_for.cpp.
 */

namespace Tests {

//! @test The @ref Kokkos::Execution::parallel_for_t operator may be used in a constant expression.
consteval bool test_parallel_for_in_constant_expression() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    [[maybe_unused]]
    const auto sndr_a = Kokkos::Execution::parallel_for(42, functor_t{});

    [[maybe_unused]]
    const auto sndr_b = Kokkos::Execution::parallel_for(Kokkos::range_slice<size_t, size_t>{42, 43}, functor_t{});

    return std::is_constant_evaluated();
}

static_assert(test_parallel_for_in_constant_expression());

class ParallelForDataTest : public testing::Test {
   public:
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using iter_space_t = Kokkos::range_slice<int, int>;

    using data_t = Kokkos::Execution::Impl::ParallelForData<functor_t, iter_space_t>;

    using labeled_data_t = Kokkos::Execution::Impl::ParallelForData<
        functor_t,
        iter_space_t,
        Kokkos::Execution::Impl::Label<std::string_view>
    >;

    using multi_data_t = Kokkos::Execution::Impl::ParallelForData<
        functor_t,
        iter_space_t,
        Kokkos::Execution::Impl::Label<std::string>,
        std::monostate
    >;
};

//! @test Check that @ref Kokkos::Execution::Impl::parallel_for_data works as expected.
consteval bool test_parallel_for_data_concept() {
    static_assert(!Kokkos::Execution::Impl::parallel_for_data<int>);
    static_assert(Kokkos::Execution::Impl::parallel_for_data<typename ParallelForDataTest::data_t>);
    static_assert(Kokkos::Execution::Impl::parallel_for_data<typename ParallelForDataTest::labeled_data_t>);
    static_assert(Kokkos::Execution::Impl::parallel_for_data<typename ParallelForDataTest::multi_data_t>);

    return true;
}

static_assert(test_parallel_for_data_concept());

//! @test Check @ref Kokkos::Execution::Impl::ParallelForData traits.
consteval bool test_parallel_for_data_traits() {
    //! Is an aggregate.
    static_assert(std::is_aggregate_v<typename ParallelForDataTest::data_t>);
    static_assert(std::is_aggregate_v<typename ParallelForDataTest::labeled_data_t>);
    static_assert(std::is_aggregate_v<typename ParallelForDataTest::multi_data_t>);

    //! Is nothrow move constructible.
    static_assert(std::is_nothrow_move_constructible_v<typename ParallelForDataTest::data_t>);
    static_assert(std::is_nothrow_move_constructible_v<typename ParallelForDataTest::labeled_data_t>);
    static_assert(std::is_nothrow_move_constructible_v<typename ParallelForDataTest::multi_data_t>);

    //! Aliases.
    static_assert(
        std::same_as<typename ParallelForDataTest::data_t::functor_t, typename ParallelForDataTest::functor_t>);
    static_assert(std::same_as<
                  typename ParallelForDataTest::data_t::policy_or_iter_space_t,
                  typename ParallelForDataTest::iter_space_t
    >);
    static_assert(
        std::same_as<typename ParallelForDataTest::data_t::properties_t, Kokkos::Execution::Impl::properties<>>);
    static_assert(std::same_as<
                  typename ParallelForDataTest::labeled_data_t::properties_t,
                  Kokkos::Execution::Impl::properties<Kokkos::Execution::Impl::Label<std::string_view>>
    >);

    //! Is labeled with @ref Kokkos::Execution::Impl::labeled.
    static_assert(!Kokkos::Execution::Impl::labeled<typename ParallelForDataTest::data_t>);
    static_assert(Kokkos::Execution::Impl::labeled<typename ParallelForDataTest::labeled_data_t>);

    return true;
}

static_assert(test_parallel_for_data_traits());

//! @test Check that @ref Kokkos::Execution::Impl::ParallelForData size depends on the given properties.
consteval bool test_parallel_for_data_size() {
    static_assert(sizeof(typename ParallelForDataTest::functor_t) == 1);
    static_assert(std::alignment_of_v<typename ParallelForDataTest::functor_t> == 1);

    static_assert(sizeof(typename ParallelForDataTest::iter_space_t) == 8);
    static_assert(std::alignment_of_v<typename ParallelForDataTest::iter_space_t> == 4);

    static_assert(std::is_empty_v<typename ParallelForDataTest::functor_t>);
    static_assert(std::is_standard_layout_v<typename ParallelForDataTest::data_t>);

    static_assert(offsetof(typename ParallelForDataTest::data_t, functor) == 0);
    static_assert(offsetof(typename ParallelForDataTest::data_t, policy_or_iter_space) == 4);

    static_assert(sizeof(typename ParallelForDataTest::data_t) == 12);
    static_assert(sizeof(typename ParallelForDataTest::labeled_data_t) == 32);

    return true;
}

static_assert(test_parallel_for_data_size());

//! @test Check that @ref Kokkos::Execution::Impl::parallel_for_data_from_t works as expected.
consteval bool test_parallel_for_data_from_t() {
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::parallel_for_data_from_t<
                      typename ParallelForDataTest::functor_t,
                      typename ParallelForDataTest::iter_space_t
                  >,
                  typename ParallelForDataTest::data_t
    >);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::parallel_for_data_from_t<
                      typename ParallelForDataTest::functor_t,
                      typename ParallelForDataTest::iter_space_t,
                      Kokkos::Execution::Impl::Label<std::string_view>
                  >,
                  typename ParallelForDataTest::labeled_data_t
    >);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::parallel_for_data_from_t<
                      typename ParallelForDataTest::functor_t,
                      typename ParallelForDataTest::iter_space_t,
                      Kokkos::Execution::Impl::Label<std::string>,
                      std::monostate
                  >,
                  typename ParallelForDataTest::multi_data_t
    >);

    return true;
}

static_assert(test_parallel_for_data_from_t());

//! @test Check that @ref Kokkos::Execution::Impl::ParallelForData::transform_policy_or_iter_space works as expected.
TEST_F(ParallelForDataTest, transform_policy_or_iter_space) {
    using labeled_data_iter_space_t = Kokkos::Execution::Impl::ParallelForData<
        functor_t,
        iter_space_t,
        Kokkos::Execution::Impl::Label<std::string_view>
    >;

    using labeled_data_range_policy_t = Kokkos::Execution::Impl::ParallelForData<
        functor_t,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>,
        Kokkos::Execution::Impl::Label<std::string_view>
    >;

    constexpr labeled_data_iter_space_t with_iter_space{
        Kokkos::Execution::Impl::Label<std::string_view>{"user"},
        functor_t{},
        iter_space_t{4, 6}
    };

    const auto with_range_policy = with_iter_space.transform_policy_or_iter_space(
        [](const auto& pol) { return Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(pol.first, pol.last); });

    static_assert(std::same_as<decltype(with_range_policy), const labeled_data_range_policy_t>);

    ASSERT_EQ(with_range_policy.policy_or_iter_space.begin(), 4);
    ASSERT_EQ(with_range_policy.policy_or_iter_space.end(), 6);
}

/**
 * @test Check that @ref Kokkos::Execution::Impl::ParallelForData::transform_policy_or_iter_space perfectly forwards.
 *
 * For an explanation of the difference between Clang and GCC, see
 * @ref Tests::ParallelReduceDataTest_transform_policy_or_iter_space_perfectly_forwards_properties_Test.
 */
TEST_F(ParallelForDataTest, transform_policy_or_iter_space_perfectly_forwards_properties) {
    using counter_t = Tests::Utils::Functors::Counter;
    using data_with_counter_t = Kokkos::Execution::Impl::ParallelForData<counter_t, iter_space_t, counter_t>;

    counter_t::reset();
    {
        data_with_counter_t from_xvalue{counter_t{}, counter_t{}, iter_space_t{}};
        [[maybe_unused]]
        const auto data_from_xvalue = std::move(from_xvalue).transform_policy_or_iter_space([](const auto&) {
            return iter_space_t{4, 6};
        });
        ASSERT_EQ(counter_t::copy_assignments, 0);
        ASSERT_EQ(counter_t::copy_constructions, 0);
        ASSERT_EQ(counter_t::move_assignments, 0);
#if defined(KOKKOS_COMPILER_CLANG) || defined(KOKKOS_COMPILER_INTEL_LLVM)
        ASSERT_EQ(counter_t::move_constructions, 3);
#else
        ASSERT_EQ(counter_t::move_constructions, 2);
#endif
    }

    counter_t::reset();
    {
        [[maybe_unused]]
        const auto data_from_prvalue = data_with_counter_t{counter_t{}, counter_t{}, iter_space_t{}}
                                           .transform_policy_or_iter_space(
                                               [](const auto&) { return iter_space_t{4, 6}; });
        ASSERT_EQ(counter_t::copy_assignments, 0);
        ASSERT_EQ(counter_t::copy_constructions, 0);
        ASSERT_EQ(counter_t::move_assignments, 0);
#if defined(KOKKOS_COMPILER_CLANG) || defined(KOKKOS_COMPILER_INTEL_LLVM)
        ASSERT_EQ(counter_t::move_constructions, 3);
#else
        ASSERT_EQ(counter_t::move_constructions, 2);
#endif
    }
}

//! @test Check that the @c noexcept contract of @ref Kokkos::Execution::parallel_for is as expected.
consteval bool test_parallel_for_noexcept_contract() {
    using label_t = std::string;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;

    static_assert(noexcept(Kokkos::Execution::parallel_for(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_for(
        std::declval<const label_t&>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_for(
        std::declval<const char*>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_for(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, true, false>&>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_for(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, true>>())));

    static_assert(noexcept(
        stdexec::just()
        | Kokkos::Execution::parallel_for(
            std::declval<label_t>(),
            std::declval<policy_t>(),
            std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    static_assert(!noexcept(
        stdexec::just()
        | Kokkos::Execution::parallel_for(
            std::declval<const label_t&>(),
            std::declval<policy_t>(),
            std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    static_assert(noexcept(Kokkos::Execution::parallel_for(
        stdexec::just(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<Kokkos::Execution::Impl::Label<std::string>>())));

    return true;
}

static_assert(test_parallel_for_noexcept_contract());

//! @test Check how @ref Kokkos::Execution::parallel_for_t deals with the label.
consteval bool test_label_storage() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    const auto clsr_literal = Kokkos::Execution::parallel_for("label", 42, functor_t{});
    const auto clsr_label = Kokkos::Execution::parallel_for(Kokkos::Execution::label("label"), 42, functor_t{});
    static_assert(std::same_as<
                  decltype(clsr_literal),
                  const stdexec::__closure<
                      Kokkos::Execution::parallel_for_t,
                      Kokkos::dims<1, int>,
                      functor_t,
                      Kokkos::Execution::Impl::Label<std::string>
                  >
    >);
    static_assert(std::same_as<
                  decltype(clsr_label),
                  const stdexec::__closure<
                      Kokkos::Execution::parallel_for_t,
                      Kokkos::dims<1, int>,
                      functor_t,
                      Kokkos::Execution::Impl::Label<std::string_view>
                  >
    >);

    return true;
}

static_assert(test_label_storage());

//! @test Check that @ref Kokkos::Execution::parallel_for_t accepts arbitrary properties.
consteval bool test_parallel_for_many_properties() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    const auto clsr_props = Kokkos::Execution::parallel_for(
        stdexec::just(),
        Kokkos::dims<1, int>{42},
        functor_t{},
        Kokkos::Execution::Impl::Label<std::string>{"my label"},
        Utils::MyPropA{"sun"},
        Utils::MyPropB{3.14});

    static_assert(std::same_as<
                  decltype(clsr_props),
                  const Kokkos::Execution::Impl::ParallelForSender<
                      stdexec::just_t::__sender<>,
                      Kokkos::Execution::Impl::ParallelForData<
                          functor_t,
                          Kokkos::dims<1, int>,
                          Kokkos::Execution::Impl::Label<std::string>,
                          Utils::MyPropA,
                          Utils::MyPropB
                      >
                  >
    >);

    return true;
}

static_assert(test_parallel_for_many_properties());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ParallelForSender into the algorithm tag, data, and child sender.
consteval bool test_parallel_for_sndr_decomposition() {
    //! @c stdexec::just sender.
    using just_sndr_t = decltype(stdexec::just());

    //! Parallel for sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_data_t = Kokkos::Execution::Impl::ParallelForData<functor_t, policy_t, label_t>;
    using pfor_sndr_t = Kokkos::Execution::Impl::ParallelForSender<just_sndr_t, pfor_data_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<pfor_sndr_t, Kokkos::Execution::parallel_for_t>);

    static_assert(std::same_as<stdexec::__data_of<pfor_sndr_t>, pfor_data_t>);

    static_assert(stdexec::__nbr_children_of<pfor_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<pfor_sndr_t>, just_sndr_t>);

    return true;
}

static_assert(test_parallel_for_sndr_decomposition());

//! @test Check @ref Kokkos::Execution::parallel_for closure object creation perfectly forwards the functor.
TEST(ParallelForTest, closure_object_creation_perfectly_forwards_functor) {
    using counter_t = Tests::Utils::Functors::Counter;

    const Kokkos::RangePolicy<TEST_EXECUTION_SPACE> policy(0, 1);
    counter_t counter;
    counter_t::reset();

    [[maybe_unused]]
    const auto closure_from_lvalue_functor = Kokkos::Execution::parallel_for("label", policy, counter);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 0);

    [[maybe_unused]]
    const auto closure_from_xvalue_functor = Kokkos::Execution::parallel_for("label", policy, std::move(counter));

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 1);

    [[maybe_unused]]
    const auto closure_from_prvalue_functor = Kokkos::Execution::parallel_for("label", policy, counter_t{});

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 2);
}

} // namespace Tests
