#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LOAD_CHECK_ADD_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LOAD_CHECK_ADD_HPP

#include "Kokkos_Core.hpp"

namespace Tests::Utils::Functors {

/**
 * @brief Load the value at @ref data and check it is equal to @ref prev. Then, add @ref value to it.
 *
 * Use @c OnDevice to check where the functor is called.
 */
template <typename ValueType, bool OnDevice>
struct LoadCheckAdd {
    ValueType prev;
    ValueType value;
    ValueType* data;

    KOKKOS_FUNCTION
    void operator()() const {
        if constexpr (OnDevice) {
            KOKKOS_IF_ON_HOST(Kokkos::abort("You should not be running on host.");)
        } else {
            KOKKOS_IF_ON_DEVICE(Kokkos::abort("You should not be running on device.");)
        }

        if (*data != prev) {
            KOKKOS_IF_ON_HOST(Kokkos::abort("Unexpected value on host.");)
            KOKKOS_IF_ON_DEVICE(Kokkos::abort("Unexpected value on device.");)
        }
        *data += value;
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LOAD_CHECK_ADD_HPP
