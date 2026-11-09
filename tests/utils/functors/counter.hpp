#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_COUNTER_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_COUNTER_HPP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

namespace Tests::Utils::Functors {

//! Count construction, move/copy construction, move/copy assignment and destruction atomically on host.
struct Counter {
    static inline std::atomic<unsigned int> default_constructions{0};
    static inline std::atomic<unsigned int> destructions{0};
    static inline std::atomic<unsigned int> copy_constructions{0};
    static inline std::atomic<unsigned int> copy_assignments{0};
    static inline std::atomic<unsigned int> move_constructions{0};
    static inline std::atomic<unsigned int> move_assignments{0};

    static inline std::atomic<unsigned int> next_id{0};

    unsigned int id{0};

    Counter() noexcept
        : id(next_id.fetch_add(1, std::memory_order_relaxed)) {
        default_constructions.fetch_add(1, std::memory_order_relaxed);
    }

    ~Counter() noexcept {
        destructions.fetch_add(1, std::memory_order_relaxed);
    }

    Counter(const Counter&) noexcept
        : id(next_id.fetch_add(1, std::memory_order_relaxed)) {
        copy_constructions.fetch_add(1, std::memory_order_relaxed);
    }

    Counter& operator=(const Counter&) noexcept {
        copy_assignments.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    Counter(Counter&&) noexcept
        : id(next_id.fetch_add(1, std::memory_order_relaxed)) {
        move_constructions.fetch_add(1, std::memory_order_relaxed);
    }

    Counter& operator=(Counter&&) noexcept {
        move_assignments.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args&&...) const noexcept {
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

//! Similar to @ref Tests::Utils::Functors::Counter, but the internal counters are not atomic.
struct CounterRef {
    struct counters {
        unsigned int constructions = 0;
        unsigned int destructions = 0;
        unsigned int copy_constructions = 0;
        unsigned int copy_assignments = 0;
        unsigned int move_constructions = 0;
        unsigned int move_assignments = 0;
    };

    counters& m_counts; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    explicit constexpr CounterRef(counters& counts) noexcept
        : m_counts(counts) {
        ++m_counts.constructions;
    }

    constexpr ~CounterRef() noexcept {
        ++m_counts.destructions;
    }

    constexpr CounterRef(const CounterRef& other) noexcept
        : m_counts(other.m_counts) {
        ++m_counts.copy_constructions;
    }

    constexpr CounterRef& operator=(const CounterRef& other) noexcept {
        m_counts = other.m_counts;
        ++m_counts.copy_assignments;
        return *this;
    }

    constexpr CounterRef(CounterRef&& other) noexcept
        : m_counts(other.m_counts) {
        ++m_counts.move_constructions;
    }

    constexpr CounterRef& operator=(CounterRef&& other) noexcept {
        m_counts = other.m_counts;
        ++m_counts.move_assignments;
        return *this;
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_COUNTER_HPP
