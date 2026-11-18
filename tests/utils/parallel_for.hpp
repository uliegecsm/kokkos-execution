#ifndef KOKKOS_EXECUTION_TESTS_UTILS_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_PARALLEL_FOR_HPP

//! @name Compose @ref Kokkos::Execution::parallel_for that use the same policy and functor.
///@{
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_AGAIN() KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_IMPL

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_BASE(_context_, _policy_, _functor_, ...)                                  \
    stdexec::schedule(_context_.get_scheduler())

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_STEP(_context_, _policy_, _functor_, _name_, ...)                          \
    Kokkos::Execution::parallel_for(                                                                                   \
        KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_AGAIN STDEXEC_PP_PARENS()(_context_, _policy_, _functor_, __VA_ARGS__),    \
        STDEXEC_PP_STRINGIZE(_name_),                                                                                  \
        _policy_,                                                                                                      \
        _functor_)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_IMPL(_context_, _policy_, _functor_, ...)                                  \
    STDEXEC_PP_IF(                                                                                                     \
        STDEXEC_PP_IS_EMPTY(__VA_ARGS__),                                                                              \
        KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_BASE,                                                                      \
        KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_STEP)                                                                      \
    (_context_, _policy_, _functor_, __VA_ARGS__)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_TESTS_PARALLEL_FOR(_context_, _policy_, _functor_, ...)                                       \
    STDEXEC_PP_EXPAND_R(KOKKOS_EXECUTION_TESTS_PARALLEL_FOR_IMPL(_context_, _policy_, _functor_, __VA_ARGS__))
///@}

#endif // KOKKOS_EXECUTION_TESTS_UTILS_PARALLEL_FOR_HPP
