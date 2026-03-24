#ifndef KOKKOS_EXECUTION_UTILS_IGNORE_WARNINGS_HPP
#define KOKKOS_EXECUTION_UTILS_IGNORE_WARNINGS_HPP

#define DO_PRAGMA_(x) _Pragma(#x)

#if defined(__clang__)
#    define PRAGMA_DIAGNOSTIC_POP           _Pragma("clang diagnostic pop")
#    define PRAGMA_DIAGNOSTIC_PUSH          _Pragma("clang diagnostic push")
//NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define PRAGMA_DIAGNOSTIC_IGNORED(what) DO_PRAGMA_(clang diagnostic ignored what)
#elif defined(__GNUC__) || defined(__GNUG__)
#    define PRAGMA_DIAGNOSTIC_POP           _Pragma("GCC diagnostic pop")
#    define PRAGMA_DIAGNOSTIC_PUSH          _Pragma("GCC diagnostic push")
#    define PRAGMA_DIAGNOSTIC_IGNORED(what) DO_PRAGMA_(GCC diagnostic ignored what)
#else
#    error "Unsupported compiler."
#endif

#if defined(__clang__)
#    define KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED_COMPILER_SPECIFIC                                       \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-attributes")
#else
#    define KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED_COMPILER_SPECIFIC
#endif

//! Basic list of ignored diagnostics when including anything from @c stdexec.
#if !defined(__DOXYGEN__)
#    define KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED                                                         \
        KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED_COMPILER_SPECIFIC                                           \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")                                                                 \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")                                                                      \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wignored-qualifiers")                                                              \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")                                                                          \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")                                                                \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")                                                                  \
        PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-result")
#else
#    define KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#endif
#endif // KOKKOS_EXECUTION_UTILS_IGNORE_WARNINGS_HPP
