#ifndef GRAPH_DISPATCHING_TESTS_UTILS_COUNTER_HPP
#define GRAPH_DISPATCHING_TESTS_UTILS_COUNTER_HPP

#include <atomic>

namespace tests::utils {

//! Count construction, move/copy construction, move/copy assignment and destruction atomically on host.
struct Counter {
    static inline std::atomic<unsigned int> default_constructions{0};
    static inline std::atomic<unsigned int> destructions{0};
    static inline std::atomic<unsigned int> copy_constructions{0};
    static inline std::atomic<unsigned int> copy_assignments{0};
    static inline std::atomic<unsigned int> move_constructions{0};
    static inline std::atomic<unsigned int> move_assignments{0};

    unsigned int id;

    Counter()
        : id(default_constructions.fetch_add(1, std::memory_order_relaxed)) {
    }

    ~Counter() {
        destructions.fetch_add(1, std::memory_order_relaxed);
    }

    Counter(const Counter&)
        : id(copy_constructions.fetch_add(1, std::memory_order_relaxed)) {
    }

    Counter& operator=(const Counter&) {
        copy_assignments.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    Counter(Counter&&) noexcept
        : id(move_constructions.fetch_add(1, std::memory_order_relaxed)) {
    }

    Counter& operator=(Counter&&) noexcept {
        move_assignments.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    void operator()() const noexcept {
    }

    static void reset() {
        default_constructions = 0;
        destructions = 0;
        copy_constructions = 0;
        copy_assignments = 0;
        move_constructions = 0;
        move_assignments = 0;
    }
};

} // namespace tests::utils

#endif // GRAPH_DISPATCHING_TESTS_UTILS_COUNTER_HPP
