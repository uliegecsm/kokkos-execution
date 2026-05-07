#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_TAG_DISPATCH_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_TAG_DISPATCH_HPP

#include <concepts>

#include "Kokkos_Macros.hpp"

namespace Tests::Utils::Functors {

template <typename ViewType>
struct TagDispatch {
    struct Tag { };

    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const Tag, const T) const noexcept {
        ++data();
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_TAG_DISPATCH_HPP
