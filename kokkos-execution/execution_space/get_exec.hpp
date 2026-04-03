#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_GET_EXEC_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_GET_EXEC_HPP

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

/**
 * Query an object for its @c Kokkos execution space instance.
 *
 * See also https://github.com/NVIDIA/cccl/blob/6e592beda9c50aeb3cc62dd1036d509f540ccbe7/libcudacxx/include/cuda/__stream/get_stream.h.
 */
struct get_exec_t : public stdexec::__query<get_exec_t> {
    using stdexec::__query<get_exec_t>::operator();
};

inline constexpr get_exec_t get_exec{};

/**
 * @brief Wrap a @c Kokkos execution space to make it cheap to copy/move in new environments.
 *
 * @warning It does not extend the lifetime of the execution space instance (it does not own it).
 *
 * Inspired by https://github.com/NVIDIA/cccl/blob/cc7c08209ed4b3ef4c80dc17fa1b8507e9d1e51f/libcudacxx/include/cuda/__stream/stream_ref.h.
 */
template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpaceRef {
    using execution_space = Exec;

    execution_space const * m_exec_ptr;

    explicit constexpr ExecutionSpaceRef(const execution_space& exec) noexcept
        : m_exec_ptr(&exec) {
    }

    ExecutionSpaceRef() = delete;
    ExecutionSpaceRef(const ExecutionSpaceRef&) noexcept = default;
    ExecutionSpaceRef& operator=(const ExecutionSpaceRef&) noexcept = default;
    ExecutionSpaceRef(ExecutionSpaceRef&&) noexcept = default;
    ExecutionSpaceRef& operator=(ExecutionSpaceRef&&) noexcept = default;
    ~ExecutionSpaceRef() noexcept = default;

    const execution_space& get() const noexcept {
        return *m_exec_ptr;
    }

    friend constexpr auto operator<=>(const ExecutionSpaceRef&, const ExecutionSpaceRef&) noexcept = default;

    [[nodiscard]]
    constexpr const ExecutionSpaceRef& query(const get_exec_t&) const noexcept {
        return *this;
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_GET_EXEC_HPP
