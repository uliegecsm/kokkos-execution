#ifndef GRAPH_DISPATCHING_TESTS_UTILS_HPP
#define GRAPH_DISPATCHING_TESTS_UTILS_HPP

#include <thread>

namespace utils
{

//! Get the current thread ID (hashed).
auto get_thread_id() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

} // namespace utils

#endif // GRAPH_DISPATCHING_TESTS_UTILS_HPP
