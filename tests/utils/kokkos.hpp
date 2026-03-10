#ifndef KOKKOS_EXECUTION_TESTS_UTILS_KOKKOS_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_KOKKOS_HPP

#include <concepts>
#include <iostream>
#include <string_view>

#include "Kokkos_Core.hpp"

namespace Tests::Utils {

template <Kokkos::ExecutionSpace Exec, Kokkos::ExecutionSpace OtherExec>
bool are_same_instances(const Exec& exec, const OtherExec& other_exec) {
    if constexpr (std::same_as<Exec, OtherExec>) {
        return exec == other_exec;
    } else {
        return false;
    }
}

template <Kokkos::ExecutionSpace Exec>
constexpr bool on_device() {
#if defined(KOKKOS_ENABLE_CUDA)
    return std::same_as<Exec, Kokkos::Cuda>;
#elif defined(KOKKOS_ENABLE_HIP)
    return std::same_as<Exec, Kokkos::HIP>;
#else
    return false;
#endif
}

template <Kokkos::ExecutionSpace Exec>
void show_exec_space_id(const Exec& exec, std::string_view label = "", std::ostream& out = std::cout) {
    out << "Execution space instance " << label << " has device ID " << Kokkos::Tools::Experimental::device_id(exec)
        << '.' << std::endl;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_KOKKOS_HPP
