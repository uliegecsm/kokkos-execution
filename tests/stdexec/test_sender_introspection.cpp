#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/no_op.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec sender introspection
 * -----------------------------------------
 *
 * This group of tests check the behavior of @c stdexec sender introspection.
 *
 * The tests can be found in @ref tests/stdexec/test_sender_introspection.cpp.
 */

namespace Tests {

using NoOp = Tests::Utils::Functors::NoOp<false, false, false>;

template <typename Child>
using then_sndr_t = stdexec::then_t::__sender<NoOp, Child>;

consteval bool test_child_of_sndr() {
    using schd_sndr_t = decltype(stdexec::schedule(std::declval<exec::single_thread_context&>().get_scheduler()));

    // __child_of<S>.
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<schd_sndr_t>>, schd_sndr_t>);
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<schd_sndr_t&>>, schd_sndr_t&>);
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<const schd_sndr_t&>>, const schd_sndr_t&>);

    // __child_of<S&>.
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<schd_sndr_t>&>, schd_sndr_t&>);
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<schd_sndr_t&>&>, schd_sndr_t&>);
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<const schd_sndr_t&>&>, const schd_sndr_t&>);

    // __child_of<const S&>.
    static_assert(std::same_as<stdexec::__child_of<const then_sndr_t<schd_sndr_t>&>, const schd_sndr_t&>);
    static_assert(std::same_as<stdexec::__child_of<const then_sndr_t<schd_sndr_t&>&>, schd_sndr_t&>);
    static_assert(std::same_as<stdexec::__child_of<const then_sndr_t<const schd_sndr_t&>&>, const schd_sndr_t&>);

    // __child_of<S&&>.
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<schd_sndr_t>&&>, schd_sndr_t&&>);
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<schd_sndr_t&>&&>, schd_sndr_t&>);
    static_assert(std::same_as<stdexec::__child_of<then_sndr_t<const schd_sndr_t&>&&>, const schd_sndr_t&>);

    //! @note The behavior corresponds to the behavior of @c stdexec::__copy_cvref_t that @c stdexec::__child_of uses internally.
    static_assert(std::same_as<stdexec::__copy_cvref_t<then_sndr_t<schd_sndr_t&>&&, schd_sndr_t&>, schd_sndr_t&>);

    //! @note The behavior corresponds to the behavior of @c stdexec::__get for @c stdexec::__tuple.
    static_assert(std::same_as<decltype(stdexec::__get<2>(std::declval<then_sndr_t<schd_sndr_t&>&&>())), schd_sndr_t&>);

    //! @note The behavior is different from the behavior of @c stdexec::__forward_like.
    static_assert(std::same_as<
                  decltype(stdexec::__forward_like<then_sndr_t<schd_sndr_t&>&&>(std::declval<schd_sndr_t&>())),
                  schd_sndr_t&&
    >);

    return true;
}

/**
 * @test Check how the cv- and reference qualifiers of a sender affect the child type given by @c stdexec::__child_of.
 *
 * - A child stored by value acquires the qualifiers of the outer sender.
 * - A child stored by lvalue reference retains its declared reference type.
 */
TEST(sender_introspection, child_of) {
    static_assert(test_child_of_sndr());
}

} // namespace Tests
