#ifndef KOKKOS_EXECUTION_TESTS_UTILS_PROPERTIES_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_PROPERTIES_HPP

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/label.hpp"

namespace Tests::Utils {

struct MyPropA {
    std::string m_value{};
};

struct MyPropB {
    double m_value = 42;
};

template <typename T>
struct LabeledData : Kokkos::Execution::Impl::Label<T> { };

template <typename T>
struct MultiPropData
    : Kokkos::Execution::Impl::Label<T>
    , MyPropA
    , MyPropB { };

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_PROPERTIES_HPP
