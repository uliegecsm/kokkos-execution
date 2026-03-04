#ifndef KOKKOS_EXECUTION_UTILS_IGNORE_WARNINGS_HPP
#define KOKKOS_EXECUTION_UTILS_IGNORE_WARNINGS_HPP

#define DO_PRAGMA_(x) _Pragma(#x)

#if defined(__clang__)
#    define PRAGMA_DIAGNOSTIC_POP  _Pragma("clang diagnostic pop")
#    define PRAGMA_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#    define PRAGMA_DIAGNOSTIC_IGNORED(what)                                                                            \
        DO_PRAGMA_(clang diagnostic ignored what) // NOLINT(cppcoreguidelines-macro-usage)
#elif defined(__GNUC__) || defined(__GNUG__)
#    define PRAGMA_DIAGNOSTIC_POP           _Pragma("GCC diagnostic pop")
#    define PRAGMA_DIAGNOSTIC_PUSH          _Pragma("GCC diagnostic push")
#    define PRAGMA_DIAGNOSTIC_IGNORED(what) DO_PRAGMA_(GCC diagnostic ignored what)
#else
#    error "Unsupported compiler."
#endif

/// We need to ignore a dangling reference warning under @c GCC, which is
/// a known issue (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107532) and a false
/// positive.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#    define PRAGMA_DIAGNOSTIC_IGNORED_DANGLING_REFERENCE PRAGMA_DIAGNOSTIC_IGNORED("-Wdangling-reference")
#else
#    define PRAGMA_DIAGNOSTIC_IGNORED_DANGLING_REFERENCE
#endif

#endif // KOKKOS_EXECUTION_UTILS_IGNORE_WARNINGS_HPP
