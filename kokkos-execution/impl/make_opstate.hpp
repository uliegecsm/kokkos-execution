#ifndef KOKKOS_EXECUTION_IMPL_MAKE_OPSTATE_HPP
#define KOKKOS_EXECUTION_IMPL_MAKE_OPSTATE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/sender_concepts.hpp"

namespace Kokkos::Execution::Impl {

template <typename DomainType, template <typename...> typename OpStateType>
struct MakeOpState {

    template <typename Sndr, typename Rcvr, typename... Clsrs>
    struct Huddle {
        using type = OpStateType<Sndr, Rcvr, Clsrs...>;

        // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
        constexpr auto operator()(Sndr&& sndr, Rcvr rcvr, Clsrs... clsrs) const
            noexcept(std::is_nothrow_constructible_v<type, Sndr&&, Rcvr&&, Clsrs&&...>) -> type {
            return type(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsrs)...);
        }
    };

    template <Impl::dispatching_sender Sndr, typename Rcvr, typename... Clsrs>
    struct Huddle<Sndr, Rcvr, Clsrs...> {
        using child_of_sndr_t = stdexec::__child_of<Sndr>;
        using clsr_of_sndr_t = typename stdexec::transform_sender_result_t<Sndr, stdexec::env_of_t<Rcvr>>::closure_t;

        using huddle_fn_t = Huddle<child_of_sndr_t, Rcvr, clsr_of_sndr_t, Clsrs...>;
        using type = typename huddle_fn_t::type;

        static constexpr bool sndr_has_nothrow_transform_sender = stdexec::__detail::__has_nothrow_transform_sender<
            DomainType,
            stdexec::set_value_t,
            Sndr&&,
            stdexec::env_of_t<Rcvr>
        >;

        static constexpr bool is_nothrow_huddle =
            std::is_nothrow_invocable_v<huddle_fn_t, child_of_sndr_t&&, Rcvr&&, clsr_of_sndr_t&&, Clsrs&&...>;

        /**
         * @note @c stdexec::__forward_like is used because @c stdexec propagates the value category
         *       of the parent sender to its child.
         *
         * See https://github.com/NVIDIA/stdexec/blob/0a3afb8de52b4fde8ca7ab62ca09a23a8aa6a30f/include/stdexec/__detail/__sender_introspection.hpp#L244-L246.
         */
        // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
        constexpr auto operator()(Sndr&& sndr, Rcvr&& rcvr, Clsrs... clsrs) const
            noexcept(sndr_has_nothrow_transform_sender && is_nothrow_huddle) -> type {
            auto trnsfrmd_sndr = stdexec::transform_sender(std::forward<Sndr>(sndr), stdexec::get_env(rcvr));
            return huddle_fn_t{}(
                stdexec::__forward_like<Sndr>(trnsfrmd_sndr.sndr),
                std::forward<Rcvr>(rcvr),
                std::move(trnsfrmd_sndr.clsr),
                std::move(clsrs)...);
        }
    };
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_MAKE_OPSTATE_HPP
