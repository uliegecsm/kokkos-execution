#ifndef GRAPH_DISPATCHING_TESTS_UTILS_HPP
#define GRAPH_DISPATCHING_TESTS_UTILS_HPP

#include <thread>

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

namespace utils
{

//! Get the current thread ID (hashed).
auto get_thread_id() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

//! Simple functor that fills a @c std::vector with the thread ID.
struct FillWithThreadID
{
    std::shared_ptr<std::vector<size_t>> ids;

    template <std::integral T>
    void operator()(const T index) {
        ids->operator[](index) = get_thread_id();
    }
};

//! Helper to define a @c then that returns the thread ID.
#define THEN_RETURN_ID ::stdexec::then([]{ return std::this_thread::get_id(); })

//! Helper to define a @c then that stores the thread ID.
#define THEN_STORE_ID(__id__, ...) ::stdexec::then([&]() -> void { __id__ = std::this_thread::get_id(); __VA_ARGS__ })

//! Helper that dumps the current thread ID to @c std::cout.
#define THEN_SHOW_ID ::stdexec::then([]() -> void { std::cout << std::this_thread::get_id() << std::endl; })

//! Helper that dumps a @c Kokkos execution space ID to @c std::cout.
#define SHOW_EXEC_SPACE_ID(_exec_) \
    std::cout << "Execution space instance " #_exec_ " has device ID " << Kokkos::Tools::Experimental::device_id(_exec_) << '.' << std::endl;

/**
 * @brief Pool of @c exec::static_thread_pool with a single thread in each of them.
 *
 * @todo Check that the list of @c IDs is unique (there is no duplicate).
 */
template <char... IDs>
class StaticThreadPool
{
private:
    template <char, typename>
    struct index_of_impl {};

    template <char ID, char... Rest>
    struct index_of_impl<ID, std::integer_sequence<char, ID, Rest...>> : std::integral_constant<std::size_t, 0> {};

    template <char ID, char Test, char... Rest>
    struct index_of_impl<ID, std::integer_sequence<char, Test, Rest...>> : std::integral_constant<std::size_t, 1 + index_of_impl<ID, std::integer_sequence<char, Rest...>>::value> {};

public:
    template <char ID>
    constexpr static auto index_of() { return index_of_impl<ID, std::integer_sequence<char, IDs...>>::value; }

    StaticThreadPool()
    {
        for(size_t ipool = 0; ipool < sizeof...(IDs); ++ipool)
            threads.at(ipool) = std::get<0>(::stdexec::sync_wait(::stdexec::schedule(pools.at(ipool).get_scheduler()) | THEN_RETURN_ID).value());
    }

protected:
    //! Ensure that all pools are initialized with a single thread.
    std::array<::exec::static_thread_pool, sizeof...(IDs)> pools {std::is_same_v<decltype(IDs), char> ...};

    std::thread::id main = std::get<0>(::stdexec::sync_wait(::stdexec::just() | THEN_RETURN_ID).value());

    std::array<std::thread::id, sizeof...(IDs)> threads;
};

} // namespace utils

#endif // GRAPH_DISPATCHING_TESTS_UTILS_HPP
