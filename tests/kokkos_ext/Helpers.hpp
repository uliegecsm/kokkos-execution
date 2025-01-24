#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP

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

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_HELPERS_HPP
