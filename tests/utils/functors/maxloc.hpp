#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_MAXLOC_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_MAXLOC_HPP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

namespace Tests::Utils::Functors {

//! Can be used in a @c parallel_reduce to find the greater of the values in @ref data alongside its index.
template <typename DataType>
struct MaxLoc {
    typename DataType::const_type data;

    template <std::integral IndexType, typename ValueType>
    KOKKOS_FUNCTION void operator()(const IndexType index, ValueType& current) const noexcept {
        if (data(index) > current.val) {
            current.val = data(index);
            current.loc = index;
        }
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_MAXLOC_HPP
