#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP

#include "Kokkos_Core.hpp"

namespace tests::kokkos_ext
{

template <typename ViewType>
struct MyDummyFunctor
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const { ++data(index); }

    template <std::integral T, typename R>
    KOKKOS_FUNCTION
    void operator()(const T index, R& current) const {
        current += data(index);
    }
};

template <typename ViewType>
struct BulkFunctor
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        Kokkos::atomic_add(data.data(), index);
    }
};

//! Get the dispatch label from @p Exec and @p label.
template <Kokkos::ExecutionSpace Exec, typename Label>
constexpr std::string dispatch_label(const Exec&, Label&& label) {
    return std::string(Kokkos::Impl::TypeInfo<Exec>::name()).append(": ").append(std::forward<Label>(label));
}

//! Add a @c then using @ref tests::ThenFunctor that may throw.
#define ADD_THEN ::stdexec::then(ThenFunctor<std::remove_cvref_t<decltype(data)>, true>{.data = data})

//! Add a @c bulk using @ref tests::kokkos_ext::BulkFunctor. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ADD_BULK(_size_) ::stdexec::bulk(::stdexec::par, _size_, BulkFunctor<std::remove_cvref_t<decltype(data)>>{.data = data})

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP
