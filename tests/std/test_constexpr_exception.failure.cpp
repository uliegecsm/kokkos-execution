#include <stdexcept>

#if !defined(__cpp_constexpr_exceptions)
#    error "C++ constexpr exception is required for this test."
#endif

namespace Tests::Std {

//! @test An exception escaping a constant-expression generates a compiler error, as stated in @cite P3068R6.
consteval bool test_escaping_exception() {
    throw std::invalid_argument{"escaping"};
}

static_assert(test_escaping_exception());

} // namespace Tests::Std
