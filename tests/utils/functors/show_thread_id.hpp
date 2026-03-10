#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_SHOW_THREAD_ID_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_SHOW_THREAD_ID_HPP

#include <iostream>
#include <thread>

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils::Functors {

struct ShowThreadID {
    void operator()() const {
        std::cout << std::this_thread::get_id() << std::endl;
    }
};

//! Add a @c then using @ref Tests::Utils::Functors::ShowThreadID. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_SHOW_THREAD_ID stdexec::then(Tests::Utils::Functors::ShowThreadID{})

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_SHOW_THREAD_ID_HPP
