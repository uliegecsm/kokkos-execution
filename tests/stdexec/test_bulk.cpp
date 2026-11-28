#include <memory>

#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c stdexec::bulk
 * --------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::bulk.
 *
 * The tests can be found in @ref tests/stdexec/test_bulk.cpp.
 */

namespace Tests {

/**
 * @test Check that a child sender is decay-copied when constructing a @c stdexec::bulk sender.
 *
 * See also:
 *  - https://eel.is/c++draft/exec#bulk-3
 *  - https://eel.is/c++draft/exec#snd.expos-24
 *  - https://eel.is/c++draft/exec#snd.expos-25
 *  - https://github.com/NVIDIA/stdexec/blob/491db342d150e17d7712deb81c5f5516c0aaab0f/include/stdexec/__detail/__basic_sender.hpp#L412-L414
 */
TEST(BulkTest, sender_construction_decay_copies_child) {
    const auto sptr = std::make_shared<int>(42);

    auto first = stdexec::just(sptr);

    ASSERT_EQ(sptr.use_count(), 2);

    const auto bulk_with_child_passed_as_const_lvalue =
        stdexec::bulk(std::as_const(first), stdexec::par, 42, [](int, const auto&) { });
    static_assert(!std::is_reference_v<
                  stdexec::__child_of<std::remove_const_t<decltype(bulk_with_child_passed_as_const_lvalue)>>
    >);
    ASSERT_EQ(sptr.use_count(), 3);

    const auto bulk_with_child_passed_as_lvalue = stdexec::bulk(first, stdexec::par, 42, [](int, const auto&) { });
    static_assert(
        !std::is_reference_v<stdexec::__child_of<std::remove_const_t<decltype(bulk_with_child_passed_as_lvalue)>>>);
    ASSERT_EQ(sptr.use_count(), 4);

    const auto bulk_with_child_passed_as_xvalue =
        stdexec::bulk(std::move(first), stdexec::par, 42, [](int, const auto&) { });
    static_assert(
        !std::is_reference_v<stdexec::__child_of<std::remove_const_t<decltype(bulk_with_child_passed_as_xvalue)>>>);
    ASSERT_EQ(sptr.use_count(), 4);
}

/**
 * @test Check that a child sender is decay-copied when transforming a @c stdexec::bulk sender.
 *
 * See also:
 *  - https://github.com/NVIDIA/stdexec/blob/5d229d49595ccd306cdd160b53ad1a018d137810/include/stdexec/__detail/__bulk.hpp#L431-L445
 */
TEST(BulkTest, sender_transformation_decay_copies_child) {
    experimental::execution::single_thread_context stc;

    const auto sptr = std::make_shared<int>(42);

    auto sndr = stdexec::just(sptr) | stdexec::continues_on(stc.get_scheduler())
              | stdexec::bulk(stdexec::par, 42, [](const int, const auto&) { });

    using sndr_t = decltype(sndr);

    ASSERT_EQ(sptr.use_count(), 2);

    const auto trnsfrmd_from_sndr_passed_as_const_lvalue =
        stdexec::transform_sender(std::as_const(sndr), stdexec::env<>{});
    using trnsfrm_const_lvalue_result_t =
        decltype(stdexec::transform_sender(std::declval<const sndr_t&>(), stdexec::env<>{}));
    static_assert(!std::is_reference_v<trnsfrm_const_lvalue_result_t>);
    static_assert(!std::same_as<trnsfrm_const_lvalue_result_t, sndr_t>);
    ASSERT_EQ(sptr.use_count(), 3);

    const auto trnsfrmd_from_sndr_passed_as_lvalue = stdexec::transform_sender(sndr, stdexec::env<>{});
    using trnsfrm_lvalue_result_t = decltype(stdexec::transform_sender(std::declval<sndr_t&>(), stdexec::env<>{}));
    static_assert(!std::is_reference_v<trnsfrm_lvalue_result_t>);
    static_assert(!std::same_as<trnsfrm_lvalue_result_t, sndr_t>);
    ASSERT_EQ(sptr.use_count(), 4);

    const auto trnsfrmd_from_sndr_passed_as_xvalue = stdexec::transform_sender(std::move(sndr), stdexec::env<>{});
    using trnsfrm_xvalue_result_t = decltype(stdexec::transform_sender(std::declval<sndr_t&&>(), stdexec::env<>{}));
    static_assert(!std::is_reference_v<trnsfrm_xvalue_result_t>);
    static_assert(!std::same_as<trnsfrm_xvalue_result_t, sndr_t>);
    ASSERT_EQ(sptr.use_count(), 4);
}

template <typename T>
struct BulkCopiesAndMovesTest : public testing::Test {
    using counter_t = Tests::Utils::Functors::Counter;

    static constexpr typename T::value_type N = T::value; // NOLINT(readability-identifier-length)

    //! @note All senders use @ref counter taken by @c lvalue.
    template <typename T::value_type M>
    stdexec::sender auto make_sndr_impl() {
        if constexpr (M == 0) {
            return stdexec::schedule(stc.get_scheduler());
        } else {
            return stdexec::bulk(make_sndr_impl<M - 1>(), stdexec::par, 42, counter);
        }
    }

    stdexec::sender auto make_sndr() {
        return make_sndr_impl<N>();
    }

    experimental::execution::single_thread_context stc;
    counter_t counter;
};

TYPED_TEST_SUITE_P(BulkCopiesAndMovesTest);

//! @test Check how many copies and moves are needed to transform a @c stdexec::bulk sender.
TYPED_TEST_P(BulkCopiesAndMovesTest, transform_sender) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr = stdexec::transform_sender(this->make_sndr(), stdexec::env<>{});

    CHECK_COUNTER_COPIES_AND_MOVES(
        EXPECT,
        0,
        this->N,
        0,
        FIX_STDEXEC_CHECK_WELL_FORMED_ADD_MOVES_TRANSFORM_BULK((this->N * (this->N + 3)) / 2 + 1))
}

//! @test Check how many copies and moves are needed to connect a @c stdexec::bulk sender.
TYPED_TEST_P(BulkCopiesAndMovesTest, connect) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const auto opstate = stdexec::connect(this->make_sndr(), Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_COPIES_AND_MOVES(
        EXPECT,
        0,
        this->N,
        0,
        FIX_STDEXEC_CHECK_WELL_FORMED_ADD_MOVES_CONNECT_BULK(
            (this->N * (this->N + 3)) / 2 + this->N * (this->N + 1) / 2 + this->N))
}

REGISTER_TYPED_TEST_SUITE_P(BulkCopiesAndMovesTest, transform_sender, connect);

using bulk_copies_and_moves_types_t = ::testing::Types<
    std::integral_constant<size_t, 1>,
    std::integral_constant<size_t, 2>,
    std::integral_constant<size_t, 3>,
    std::integral_constant<size_t, 4>,
    std::integral_constant<size_t, 5>,
    std::integral_constant<size_t, 6>,
    std::integral_constant<size_t, 9>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    BulkCopiesAndMovesTestInstantiation,
    BulkCopiesAndMovesTest,
    bulk_copies_and_moves_types_t);

} // namespace Tests
