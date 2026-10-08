#ifndef KOKKOS_EXECUTION_TESTS_UTILS_RETURN_THREAD_ID_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_RETURN_THREAD_ID_HPP

#include <thread>

#include "kokkos-execution/stdexec.hpp"

//! Return the thread ID.
#define THEN_RETURN_THREAD_ID stdexec::then([]() noexcept { return std::this_thread::get_id(); })

#endif // KOKKOS_EXECUTION_TESTS_UTILS_RETURN_THREAD_ID_HPP
