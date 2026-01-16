#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_COMPLETION_SIGNATURES_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_COMPLETION_SIGNATURES_HPP

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::impl {

//! As required by https://github.com/NVIDIA/stdexec/commit/a0d95e90fc188f4f73328c4274551434edba3165, that follows from @cite P3557R3.
#define GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(_type_)                                                     \
    template <::stdexec::__decays_to<_type_> Self, typename... Env>                                                    \
    static consteval auto get_completion_signatures() -> _completion_signatures<Self, Env...> {                        \
        return {};                                                                                                     \
    }

} // namespace Kokkos::Experimental::details::impl

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_COMPLETION_SIGNATURES_HPP
