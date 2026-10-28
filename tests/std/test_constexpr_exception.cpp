#include <stdexcept>

/**
 * @addtogroup unittests
 *
 * @c constexpr exceptions in C++26
 * --------------------------------
 *
 * This group of tests check the behavior of C++26 @c constexpr exceptions,
 * standardized in @cite P3068R6.
 *
 * The tests can be found in @ref tests/std/test_constexpr_exception.cpp.
 */

#if !defined(__cpp_constexpr_exceptions)
#    error "C++ constexpr exception is required for this test."
#endif

namespace Tests::Std {

//! Unconditionally throw a @c std::invalid_argument exception.
consteval void throw_invalid_argument() {
    throw std::invalid_argument{"message"};
}

//! @test Catching an exception in @c consteval works.
consteval bool test_catch_exception() {
    try {
        throw_invalid_argument();
        return false;
    } catch (const std::invalid_argument& iarg) {
        return std::string_view{iarg.what()} == "message";
    }
}

static_assert(test_catch_exception());

} // namespace Tests::Std
