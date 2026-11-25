#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c stdexec::__tuple
 * -----------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::__tuple.
 *
 * The tests can be found in @ref tests/stdexec/test_tuple.cpp.
 */

namespace Tests {

//! @test Building a @c stdexec::__tuple from @c prvalue elides move constructions. See also https://en.cppreference.com/cpp/language/copy_elision.
consteval bool test_stdexec_tuple_copies_and_moves() {
    using counter_t = Tests::Utils::Functors::CounterRef;

    {
        counter_t::counters counts{};

        [[maybe_unused]]
        const stdexec::__tuple six{
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts}};

        static_assert(std::is_aggregate_v<std::remove_cvref_t<decltype(six)>>);

        if (counts != counter_t::counters{6, 0, 0, 0, 0, 0})
            return false;
    }

    //! Using @c stdexec::__mktuple prevents the elision.
    {
        counter_t::counters counts{};

        [[maybe_unused]]
        const auto mk_six = stdexec::__mktuple(
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts});

        static_assert(std::is_aggregate_v<std::remove_cvref_t<decltype(mk_six)>>);

        if (counts != counter_t::counters{6, 6, 0, 0, 6, 0})
            return false;
    }

    //! @c stdexec::__tuple is specialised up to size 8, yet it allows elision also for more than 8 elements.
    {
        counter_t::counters counts{};

        PRAGMA_DIAGNOSTIC_PUSH
        PRAGMA_DIAGNOSTIC_IGNORED("-Wmissing-braces")
        [[maybe_unused]]
        const stdexec::__tuple ten{
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts}};
        PRAGMA_DIAGNOSTIC_POP

        static_assert(std::is_aggregate_v<std::remove_cvref_t<decltype(ten)>>);

        static_assert(std::derived_from<std::remove_cvref_t<decltype(ten)>, stdexec::__tup::__box<counter_t, 9>>);

        if (counts != counter_t::counters{10, 0, 0, 0, 0, 0})
            return false;
    }

    return true;
}

static_assert(test_stdexec_tuple_copies_and_moves());

//! @test @c std::tuple does not behave like @c stdexec::__tuple (it is not an aggregate, and it does not allow elision).
consteval bool test_std_tuple_copies_and_moves() {
    using counter_t = Tests::Utils::Functors::CounterRef;

    {
        counter_t::counters counts{};

        [[maybe_unused]]
        const std::tuple six{
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts}};

        static_assert(!std::is_aggregate_v<std::remove_cvref_t<decltype(six)>>);

        if (counts != counter_t::counters{6, 6, 0, 0, 6, 0})
            return false;
    }

    {
        counter_t::counters counts{};

        [[maybe_unused]]
        const auto mk_six = std::make_tuple(
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts});

        static_assert(!std::is_aggregate_v<std::remove_cvref_t<decltype(mk_six)>>);

        if (counts != counter_t::counters{6, 6, 0, 0, 6, 0})
            return false;
    }

    {
        counter_t::counters counts{};

        [[maybe_unused]]
        const std::tuple ten{
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts},
            counter_t{counts}};

        static_assert(!std::is_aggregate_v<std::remove_cvref_t<decltype(ten)>>);

        if (counts != counter_t::counters{10, 10, 0, 0, 10, 0})
            return false;
    }

    return true;
}

static_assert(test_std_tuple_copies_and_moves());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests
