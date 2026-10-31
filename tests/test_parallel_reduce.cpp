#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/parallel_reduce.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/properties.hpp"

/**
 * @addtogroup unittests
 *
 * Sender adaptor @c Kokkos::Execution::parallel_reduce
 * ----------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::parallel_reduce.
 *
 * The tests can be found in @ref tests/test_parallel_reduce.cpp.
 */

namespace Tests {

//! @test The @ref Kokkos::Execution::parallel_reduce_t operator may be used in a constant expression.
consteval bool test_parallel_reduce_in_constant_expression() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    [[maybe_unused]]
    const auto sndr_a = Kokkos::Execution::parallel_reduce(42, functor_t{}, functor_t{});

    [[maybe_unused]]
    const auto sndr_b =
        Kokkos::Execution::parallel_reduce(Kokkos::range_slice<size_t, size_t>{42, 43}, functor_t{}, functor_t{});

    return std::is_constant_evaluated();
}

static_assert(test_parallel_reduce_in_constant_expression());

class ParallelReduceDataTest : public testing::Test {
   public:
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using iter_space_t = Kokkos::range_slice<int, int>;
    using reducer_t = functor_t;

    using data_t = Kokkos::Execution::Impl::ParallelReduceData<functor_t, iter_space_t, reducer_t>;

    using labeled_data_t = Kokkos::Execution::Impl::ParallelReduceData<
        functor_t,
        iter_space_t,
        reducer_t,
        Kokkos::Execution::Impl::Label<std::string_view>
    >;

    using multi_data_t = Kokkos::Execution::Impl::ParallelReduceData<
        functor_t,
        iter_space_t,
        reducer_t,
        Kokkos::Execution::Impl::Label<std::string>,
        std::monostate
    >;
};

//! @test Check that @ref Kokkos::Execution::Impl::parallel_reduce_data works as expected.
consteval bool test_parallel_reduce_data_concept() {
    static_assert(!Kokkos::Execution::Impl::parallel_reduce_data<int>);
    static_assert(Kokkos::Execution::Impl::parallel_reduce_data<typename ParallelReduceDataTest::data_t>);
    static_assert(Kokkos::Execution::Impl::parallel_reduce_data<typename ParallelReduceDataTest::labeled_data_t>);
    static_assert(Kokkos::Execution::Impl::parallel_reduce_data<typename ParallelReduceDataTest::multi_data_t>);

    return true;
}

static_assert(test_parallel_reduce_data_concept());

//! @test Check @ref Kokkos::Execution::Impl::ParallelReduceData traits.
consteval bool test_parallel_reduce_data_traits() {
    //! Is an aggregate.
    static_assert(std::is_aggregate_v<typename ParallelReduceDataTest::data_t>);
    static_assert(std::is_aggregate_v<typename ParallelReduceDataTest::labeled_data_t>);
    static_assert(std::is_aggregate_v<typename ParallelReduceDataTest::multi_data_t>);

    //! Is nothrow move constructible.
    static_assert(std::is_nothrow_move_constructible_v<typename ParallelReduceDataTest::data_t>);
    static_assert(std::is_nothrow_move_constructible_v<typename ParallelReduceDataTest::labeled_data_t>);
    static_assert(std::is_nothrow_move_constructible_v<typename ParallelReduceDataTest::multi_data_t>);

    //! Aliases.
    static_assert(
        std::same_as<typename ParallelReduceDataTest::data_t::functor_t, typename ParallelReduceDataTest::functor_t>);
    static_assert(std::same_as<
                  typename ParallelReduceDataTest::data_t::policy_or_iter_space_t,
                  typename ParallelReduceDataTest::iter_space_t
    >);
    static_assert(
        std::same_as<typename ParallelReduceDataTest::data_t::reducer_t, typename ParallelReduceDataTest::reducer_t>);
    static_assert(
        std::same_as<typename ParallelReduceDataTest::data_t::properties_t, Kokkos::Execution::Impl::properties<>>);
    static_assert(std::same_as<
                  typename ParallelReduceDataTest::labeled_data_t::properties_t,
                  Kokkos::Execution::Impl::properties<Kokkos::Execution::Impl::Label<std::string_view>>
    >);

    //! Is labeled with @ref Kokkos::Execution::Impl::labeled.
    static_assert(!Kokkos::Execution::Impl::labeled<typename ParallelReduceDataTest::data_t>);
    static_assert(Kokkos::Execution::Impl::labeled<typename ParallelReduceDataTest::labeled_data_t>);

    return true;
}

static_assert(test_parallel_reduce_data_traits());

//! @test Check that @ref Kokkos::Execution::Impl::ParallelReduceData size depends on the given properties.
consteval bool test_parallel_reduce_data_size() {
    static_assert(sizeof(typename ParallelReduceDataTest::functor_t) == 1);
    static_assert(std::alignment_of_v<typename ParallelReduceDataTest::functor_t> == 1);

    static_assert(sizeof(typename ParallelReduceDataTest::iter_space_t) == 8);
    static_assert(std::alignment_of_v<typename ParallelReduceDataTest::iter_space_t> == 4);

    static_assert(sizeof(typename ParallelReduceDataTest::reducer_t) == 1);
    static_assert(std::alignment_of_v<typename ParallelReduceDataTest::reducer_t> == 1);

    static_assert(std::is_empty_v<typename ParallelReduceDataTest::functor_t>);
    static_assert(std::is_standard_layout_v<typename ParallelReduceDataTest::data_t>);
    static_assert(std::is_empty_v<typename ParallelReduceDataTest::reducer_t>);

    static_assert(offsetof(typename ParallelReduceDataTest::data_t, functor) == 0);
    static_assert(offsetof(typename ParallelReduceDataTest::data_t, policy_or_iter_space) == 4);
    static_assert(offsetof(typename ParallelReduceDataTest::data_t, reducer) == 12);

    static_assert(sizeof(typename ParallelReduceDataTest::data_t) == 16);
    static_assert(sizeof(typename ParallelReduceDataTest::labeled_data_t) == 32);

    return true;
}

static_assert(test_parallel_reduce_data_size());

//! @test Check that @ref Kokkos::Execution::Impl::parallel_reduce_data_from_t works as expected.
consteval bool test_parallel_reduce_data_from_t() {
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::parallel_reduce_data_from_t<
                      typename ParallelReduceDataTest::functor_t,
                      typename ParallelReduceDataTest::iter_space_t,
                      typename ParallelReduceDataTest::reducer_t
                  >,
                  typename ParallelReduceDataTest::data_t
    >);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::parallel_reduce_data_from_t<
                      typename ParallelReduceDataTest::functor_t,
                      typename ParallelReduceDataTest::iter_space_t,
                      typename ParallelReduceDataTest::reducer_t,
                      Kokkos::Execution::Impl::Label<std::string_view>
                  >,
                  typename ParallelReduceDataTest::labeled_data_t
    >);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::parallel_reduce_data_from_t<
                      typename ParallelReduceDataTest::functor_t,
                      typename ParallelReduceDataTest::iter_space_t,
                      typename ParallelReduceDataTest::reducer_t,
                      Kokkos::Execution::Impl::Label<std::string>,
                      std::monostate
                  >,
                  typename ParallelReduceDataTest::multi_data_t
    >);

    return true;
}

static_assert(test_parallel_reduce_data_from_t());

//! @test Check that @ref Kokkos::Execution::Impl::ParallelReduceData::transform_policy_or_iter_space works as expected.
TEST_F(ParallelReduceDataTest, transform_policy_or_iter_space) {
    using labeled_data_iter_space_t = Kokkos::Execution::Impl::ParallelReduceData<
        functor_t,
        iter_space_t,
        reducer_t,
        Kokkos::Execution::Impl::Label<std::string_view>
    >;

    using labeled_data_range_policy_t = Kokkos::Execution::Impl::ParallelReduceData<
        functor_t,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>,
        reducer_t,
        Kokkos::Execution::Impl::Label<std::string_view>
    >;

    constexpr labeled_data_iter_space_t with_iter_space{
        Kokkos::Execution::Impl::Label<std::string_view>{"user"},
        functor_t{},
        iter_space_t{4, 6},
        reducer_t{}
    };

    const auto with_range_policy = with_iter_space.transform_policy_or_iter_space(
        [](const auto& pol) { return Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(pol.first, pol.last); });

    static_assert(std::same_as<decltype(with_range_policy), const labeled_data_range_policy_t>);

    ASSERT_EQ(with_range_policy.policy_or_iter_space.begin(), 4);
    ASSERT_EQ(with_range_policy.policy_or_iter_space.end(), 6);
}

/**
 * @test Check that @ref Kokkos::Execution::Impl::ParallelReduceData::transform_policy_or_iter_space perfectly forwards.
 *
 * Note that this test will use @ref Tests::Utils::Functors::Counter both as a member of
 * @ref Kokkos::Execution::Impl::ParallelReduceData, but also as a property (in which case
 * it becomes a base class of @ref Kokkos::Execution::Impl::ParallelReduceData).
 *
 * Copy elision is applied on the members, but not on base classes, see
 * https://github.com/llvm/llvm-project/issues/33864. While Clang is conforming, GCC is not.
 */
TEST_F(ParallelReduceDataTest, transform_policy_or_iter_space_perfectly_forwards_properties) {
    using counter_t = Tests::Utils::Functors::Counter;
    using data_with_counter_t =
        Kokkos::Execution::Impl::ParallelReduceData<counter_t, iter_space_t, reducer_t, counter_t>;

    counter_t::reset();
    {
        data_with_counter_t from_xvalue{counter_t{}, counter_t{}, iter_space_t{}, reducer_t{}};
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
        const auto data_from_prvalue = data_with_counter_t{counter_t{}, counter_t{}, iter_space_t{}, reducer_t{}}
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

//! @test Check that the @c noexcept contract of @ref Kokkos::Execution::parallel_reduce is as expected.
consteval bool test_parallel_reduce_noexcept_contract() {
    using label_t = std::string;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = Kokkos::View<int, TEST_EXECUTION_SPACE>;

    static_assert(noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<reducer_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<const label_t&>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<reducer_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<const char*>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<reducer_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, true, false>&>(),
        std::declval<reducer_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, true>>(),
        std::declval<reducer_t>())));

    static_assert(noexcept(
        stdexec::just()
        | Kokkos::Execution::parallel_reduce(
            std::declval<label_t>(),
            std::declval<policy_t>(),
            std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
            std::declval<reducer_t>())));

    static_assert(!noexcept(
        stdexec::just()
        | Kokkos::Execution::parallel_reduce(
            std::declval<const label_t&>(),
            std::declval<policy_t>(),
            std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
            std::declval<reducer_t>())));

    //! With @ref Kokkos::Execution::Impl::Label directly.
    using label_of_string_t = Kokkos::Execution::Impl::Label<std::string>;

    static_assert(noexcept(Kokkos::Execution::parallel_reduce(
        stdexec::just(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<reducer_t>(),
        std::declval<label_of_string_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        stdexec::just(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<reducer_t>(),
        std::declval<const label_of_string_t&>())));

    return true;
}

static_assert(test_parallel_reduce_noexcept_contract());

//! @test Check how @ref Kokkos::Execution::parallel_reduce_t deals with the label.
consteval bool test_label_storage() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    const auto clsr_literal = Kokkos::Execution::parallel_reduce("label", 42, functor_t{}, functor_t{});
    const auto clsr_label =
        Kokkos::Execution::parallel_reduce(Kokkos::Execution::label("label"), 42, functor_t{}, functor_t{});
    static_assert(std::same_as<
                  decltype(clsr_literal),
                  const stdexec::__closure<
                      Kokkos::Execution::parallel_reduce_t,
                      Kokkos::dims<1, int>,
                      functor_t,
                      functor_t,
                      Kokkos::Execution::Impl::Label<std::string>
                  >
    >);
    static_assert(std::same_as<
                  decltype(clsr_label),
                  const stdexec::__closure<
                      Kokkos::Execution::parallel_reduce_t,
                      Kokkos::dims<1, int>,
                      functor_t,
                      functor_t,
                      Kokkos::Execution::Impl::Label<std::string_view>
                  >
    >);

    return true;
}

static_assert(test_label_storage());

//! @test Check that @ref Kokkos::Execution::parallel_reduce_t accepts arbitrary properties.
consteval bool test_parallel_reduce_many_properties() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using reducer_t = functor_t;

    const auto clsr_props = Kokkos::Execution::parallel_reduce(
        stdexec::just(),
        Kokkos::dims<1, int>{42},
        functor_t{},
        reducer_t{},
        Kokkos::Execution::Impl::Label<std::string>{"my label"},
        Utils::MyPropA{"sun"},
        Utils::MyPropB{3.14});

    static_assert(std::same_as<
                  decltype(clsr_props),
                  const Kokkos::Execution::Impl::ParallelReduceSender<
                      stdexec::just_t::__sender<>,
                      Kokkos::Execution::Impl::ParallelReduceData<
                          functor_t,
                          Kokkos::dims<1, int>,
                          reducer_t,
                          Kokkos::Execution::Impl::Label<std::string>,
                          Utils::MyPropA,
                          Utils::MyPropB
                      >
                  >
    >);

    return true;
}

static_assert(test_parallel_reduce_many_properties());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ParallelReduceSender into the algorithm tag, data, and child sender.
consteval bool test_parallel_reduce_sndr_decomposition() {
    //! @c stdexec::just sender.
    using just_sndr_t = decltype(stdexec::just());

    //! Parallel reducer sender.
    using label_t = Kokkos::Execution::Impl::Label<std::string>;
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = Kokkos::View<int, TEST_EXECUTION_SPACE>;
    using pred_data_t = Kokkos::Execution::Impl::ParallelReduceData<functor_t, policy_t, reducer_t, label_t>;
    using pred_sndr_t = Kokkos::Execution::Impl::ParallelReduceSender<just_sndr_t, pred_data_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<pred_sndr_t, Kokkos::Execution::parallel_reduce_t>);

    static_assert(std::same_as<stdexec::__data_of<pred_sndr_t>, pred_data_t>);

    static_assert(stdexec::__nbr_children_of<pred_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<pred_sndr_t>, just_sndr_t>);

    return true;
}

static_assert(test_parallel_reduce_sndr_decomposition());

//! @test Check @ref Kokkos::Execution::parallel_reduce closure object creation perfectly forwards the functor.
TEST(ParallelReduceTest, closure_object_creation_perfectly_forwards_functor) {
    using counter_t = Tests::Utils::Functors::Counter;

    const Kokkos::RangePolicy<TEST_EXECUTION_SPACE> policy(0, 1);
    const Kokkos::View<int, TEST_EXECUTION_SPACE> reducer;
    counter_t counter;
    counter_t::reset();

    [[maybe_unused]]
    const auto closure_from_lvalue_functor = Kokkos::Execution::parallel_reduce("label", policy, counter, reducer);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 0);

    [[maybe_unused]]
    const auto closure_from_xvalue_functor =
        Kokkos::Execution::parallel_reduce("label", policy, std::move(counter), reducer);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 1);

    [[maybe_unused]]
    const auto closure_from_prvalue_functor = Kokkos::Execution::parallel_reduce("label", policy, counter_t{}, reducer);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 2);
}

} // namespace Tests
