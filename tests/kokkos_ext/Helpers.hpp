#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP

#include "Kokkos_Core.hpp"

#include "kokkos-utils/concepts/View.hpp"

namespace tests::kokkos_ext {

template <typename ViewType>
struct MyDummyFunctor {
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const {
        ++data(index);
    }

    template <std::integral T, typename R>
    KOKKOS_FUNCTION void operator()(const T index, R& current) const {
        current += data(index);
    }
};

template <typename ViewType>
struct BulkFunctor {
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const {
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

template <Kokkos::utils::concepts::View ViewType>
using atomic = Kokkos::View<
    typename ViewType::traits::data_type,
    typename ViewType::traits::array_layout,
    typename ViewType::traits::device_type,
    typename ViewType::traits::hooks_policy,
    Kokkos::MemoryTraits<ViewType::traits::memory_traits::impl_value | Kokkos::Atomic>
>;

//! Same as @ref ADD_THEN with atomic memory traits.
#define ADD_THEN_ATOMIC ::stdexec::then(ThenFunctor<atomic<std::remove_cvref_t<decltype(data)>>, true>{.data = data})

//! Add a @c bulk using @ref tests::kokkos_ext::BulkFunctor. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ADD_BULK(_size_)                                                                                               \
    ::stdexec::bulk(::stdexec::par, _size_, BulkFunctor<std::remove_cvref_t<decltype(data)>>{.data = data})

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP
