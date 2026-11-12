#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LIFETIME_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LIFETIME_HPP

#include <atomic>

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core_impl;
#else
#    include "impl/Kokkos_SharedAlloc.hpp"
#endif

#include "kokkos-execution/impl/immovable.hpp"

namespace Tests::Utils::Functors {

class LifetimeSource final : Kokkos::Impl::SharedAllocationRecord<void, void> {
   private:
    using base_t = Kokkos::Impl::SharedAllocationRecord<void, void>;

    friend struct Lifetime;

    std::atomic<unsigned int> m_high_water{0};

    //! Record the current count as a new peak if it is one.
    void sample() noexcept {
        const auto current = use_count();
        unsigned int previous = m_high_water.load(std::memory_order_relaxed);
        while (previous < current
               && !m_high_water.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
        }
    }

    //! Sample the source behind @p tracker, if it is tracking one.
    static void sample_from(const Kokkos::Impl::SharedAllocationTracker& tracker) noexcept {
        if (base_t* const record = tracker.get_record<void>())
            static_cast<LifetimeSource*>(record)->sample();
    }

   public:
    LifetimeSource() {
        base_t::increment(this);
    }

    [[nodiscard]]
    base_t* get() noexcept {
        return this;
    }

    //! Live, tracking-enabled trackers.
    [[nodiscard]]
    unsigned int use_count() const noexcept {
        return base_t::use_count() - 1;
    }

    //! Highest @ref use_count observed through @ref Lifetime copies since construction or the last reset.
    [[nodiscard]]
    unsigned int high_water() const noexcept {
        return m_high_water.load(std::memory_order_relaxed);
    }

    //! Restart from the current count (not from zero).
    void reset_high_water() noexcept {
        m_high_water.store(use_count(), std::memory_order_relaxed);
    }
};

struct Lifetime {
    using tracker_t = Kokkos::Impl::SharedAllocationTracker;

    tracker_t tracker;

    Lifetime() noexcept = default;

    explicit Lifetime(LifetimeSource& source) noexcept {
        tracker.assign_allocated_record_to_uninitialized(source.get());
        LifetimeSource::sample_from(tracker);
    }

    KOKKOS_FUNCTION Lifetime(const Lifetime& rhs) noexcept
        : tracker(rhs.tracker) {
        KOKKOS_IF_ON_HOST((LifetimeSource::sample_from(tracker);))
    }

    KOKKOS_FUNCTION Lifetime& operator=(const Lifetime& rhs) noexcept {
        tracker = rhs.tracker;
        KOKKOS_IF_ON_HOST((LifetimeSource::sample_from(tracker);))
        return *this;
    }

    Lifetime(Lifetime&&) noexcept = default;
    Lifetime& operator=(Lifetime&&) noexcept = default;

    ~Lifetime() = default;

    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args&&...) const noexcept {
    }
};
} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LIFETIME_HPP
