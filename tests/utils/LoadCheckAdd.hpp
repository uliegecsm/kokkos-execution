#ifndef GRAPH_DISPATCHING_TESTS_UTILS_LOADCHECKADD_HPP
#define GRAPH_DISPATCHING_TESTS_UTILS_LOADCHECKADD_HPP

#include "Kokkos_Core.hpp"

namespace tests::utils {
/**
 * @brief Load the value at @ref data and check it is equal to @ref prev. Then, add @ref value to it.
 *
 * Use @c OnDevice to check where the functor is called.
 */
template <typename ValueType, bool OnDevice>
struct LoadCheckAddFunctor {
    ValueType prev;
    ValueType value;
    ValueType* data;

    KOKKOS_FUNCTION
    void operator()() const {
        if constexpr (OnDevice) {
            KOKKOS_IF_ON_HOST(Kokkos::abort("Bulk: you should not be running on host.");)
        } else {
            KOKKOS_IF_ON_DEVICE(Kokkos::abort("Bulk: you should not be running on device.");)
        }

        if (*data != prev)
            Kokkos::abort("Unexpected value.");
        *data += value;
    }
};

} // namespace tests::utils

#endif // GRAPH_DISPATCHING_TESTS_UTILS_LOADCHECKADD_HPP
