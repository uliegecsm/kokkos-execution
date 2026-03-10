#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_SUM_INDICES_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_SUM_INDICES_HPP

#include <concepts>
#include <type_traits>

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Tests::Utils::Functors {

template <typename ViewType>
struct SumIndices {
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const {
        Kokkos::atomic_add(data.data(), index);
    }

    template <Kokkos::TeamHandle TeamHandleType>
    KOKKOS_FUNCTION void operator()(const TeamHandleType& team_handle) const {
        const auto start_index = team_handle.league_rank() * team_handle.team_size();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_handle, team_handle.team_size()),
            [&]<std::integral T>(const T index) { Kokkos::atomic_add(data.data(), start_index + index); });
    }
};

//! Add a @c bulk using @ref Tests::Utils::Functors::SumIndices. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BULK_SUM_INDICES(_size_, _data_)                                                                               \
    stdexec::bulk(                                                                                                     \
        stdexec::par,                                                                                                  \
        _size_,                                                                                                        \
        Tests::Utils::Functors::SumIndices<std::remove_cvref_t<decltype(_data_)>>{.data = _data_})

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_SUM_INDICES_HPP
