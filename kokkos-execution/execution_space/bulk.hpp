#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/diagnostics.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/bulk.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct TransformSenderFor<stdexec::bulk_t> {
    template <typename Env, typename Data, typename Sndr>
    struct Transformer {
        using execution_space = Impl::exec_of_t<Sndr, Env>;

        using functor_t = Impl::functor_for_bulk_t<Data>;
        using iter_space_t = Impl::iter_space_for_bulk_t<Data>;

        using data_t = Impl::ParallelForData<functor_t, iter_space_t, Impl::Label<std::string_view>>;

        using sndr_t = ParallelForSender<stdexec::bulk_t, Sndr, execution_space, data_t>;

        auto operator()(const Env& env, std::remove_reference_t<Data>& data, std::remove_reference_t<Sndr>& sndr) const
            STDEXEC_AUTO_RETURN(
                sndr_t{
                    {.state = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                                  .state,
                     .data =
                         {{Impl::Label{KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(execution_space, ": bulk")}},
                          stdexec::__forward_like<Data>(data.__fun_),
                          iter_space_t(stdexec::__forward_like<Data>(data.__shape_))}},
                    std::forward<Sndr>(sndr)
        })
    };

    template <typename Env, Impl::has_parallel_policy Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr  // NOLINT(cppcoreguidelines-missing-std-forward)
    ) const STDEXEC_AUTO_RETURN(Transformer<Env, Data, Sndr>{}(env, data, sndr))

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::bulk_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
