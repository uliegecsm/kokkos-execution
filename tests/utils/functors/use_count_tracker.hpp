#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_USECOUNTTRACKER_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_USECOUNTTRACKER_HPP

#include "impl/Kokkos_HostSharedPtr.hpp"

#include "kokkos-execution/impl/immovable.hpp"

namespace Tests::Utils::Functors {

//! Use count tracking source, to be used with @ref UseCountTracker.
class UseCountTrackerSource : public Kokkos::Execution::Impl::Immovable {
   public:
    struct State {
        std::atomic<unsigned int> m_high_water{0};

        //! Record @p current as a new peak if it is one.
        void sample(const unsigned int current) noexcept {
            unsigned int previous = m_high_water.load(std::memory_order_relaxed);
            while (previous < current
                   && !m_high_water.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {
            }
        }
    };

    using state_t = Kokkos::Impl::HostSharedPtr<State>;

    UseCountTrackerSource()
        : m_state(new State) {
    }

    [[nodiscard]]
    constexpr const state_t& get_state() const noexcept {
        return m_state;
    }

    //! Live @ref UseCountTracker objects holding a reference.
    [[nodiscard]]
    unsigned int use_count() const noexcept {
        return static_cast<unsigned int>(m_state.use_count() - 1);
    }

    //! Highest @ref use_count observed through @ref UseCountTracker copies since construction or the last call to @ref reset_high_water.
    [[nodiscard]]
    unsigned int high_water() const noexcept {
        return m_state.get()->m_high_water.load(std::memory_order_relaxed);
    }

    //! Restart from the current count.
    void reset_high_water() noexcept {
        m_state.get()->m_high_water.store(use_count(), std::memory_order_relaxed);
    }

   private:
    state_t m_state;
};

struct UseCountTracker {
    UseCountTrackerSource::state_t m_state;

    constexpr UseCountTracker() noexcept = default;

    explicit UseCountTracker(const UseCountTrackerSource& source) noexcept
        : m_state(source.get_state()) {
        sample();
    }

    KOKKOS_FUNCTION UseCountTracker(const UseCountTracker& rhs) noexcept
        : m_state(rhs.m_state) {
        KOKKOS_IF_ON_HOST(sample();)
    }

    KOKKOS_FUNCTION UseCountTracker& operator=(const UseCountTracker& rhs) noexcept {
        this->m_state = rhs.m_state;
        KOKKOS_IF_ON_HOST(sample();)
        return *this;
    }

    UseCountTracker(UseCountTracker&&) noexcept = default;
    UseCountTracker& operator=(UseCountTracker&&) noexcept = default;
    ~UseCountTracker() = default;

    [[nodiscard]]
    bool tracked() const noexcept {
        return m_state.use_count() > 0;
    }

    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args&&...) const noexcept {
    }

   private:
    void sample() const noexcept {
        if (const unsigned int total = m_state.use_count(); total > 0)
            if (UseCountTrackerSource::State* const ptr = m_state.get())
                ptr->sample(total - 1);
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_USECOUNTTRACKER_HPP
