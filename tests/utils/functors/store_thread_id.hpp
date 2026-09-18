#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_STORE_THREAD_ID_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_STORE_THREAD_ID_HPP

#include <thread>

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils::Functors {

struct StoreThreadID {
    void operator()() const {
        *tid = std::this_thread::get_id();
    }

    std::thread::id* tid;
};

//! Add a @c then calling @ref Tests::Utils::Functors::StoreThreadID. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_STORE_THREAD_ID(__id__) stdexec::then(Tests::Utils::Functors::StoreThreadID{__id__})

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_STORE_THREAD_ID_HPP
