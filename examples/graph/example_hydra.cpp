#include "kokkos-execution/graph.hpp"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

/**
 * @addtogroup examples
 *
 * @example examples/graph/example_hydra.cpp
 *
 * A @c stdexec::when_all with a configurable number of branches in a @c Kokkos::Graph
 * -----------------------------------------------------------------------------------
 *
 * This example creates a @c stdexec::when_all with one branch per character of a string
 * (see @ref Examples::Graph::Hydra::pattern).
 * Each branch (a "head" of the hydra) writes its own character to its own slot of a view.
 * Cut off a head and two grow back: to add a branch, just add a character to the pattern,
 * no other change needed.
 */

namespace Examples::Graph {

template <typename ViewType, char Value, std::size_t Index>
struct Write {
    ViewType data;

    KOKKOS_FUNCTION void operator()() const noexcept {
        data(Index) = Value;
    }
};

template <Kokkos::ExecutionSpace Exec>
struct Hydra : public Kokkos::utils::tests::scoped::ExecutionSpace<Exec> {
   public:
    using base_t = Kokkos::utils::tests::scoped::ExecutionSpace<Exec>;

    //! One branch (head) per character. To grow another head, extend the string.
    static constexpr std::string_view pattern = "SORRY HERACLES";

    //! One extra slot, zero-initialized, so the result is always null-terminated.
    using view_t = Kokkos::View<char[pattern.size() + 1], Exec>;

   public:
    Hydra()
        : base_t()
        , data(Kokkos::view_alloc(this->exec, "data")) {
    }

    void check() const {
        const auto mirror = Kokkos::create_mirror_view_and_copy(data);
        const std::string_view got{mirror.data(), pattern.size()};

        std::printf( // NOLINT(modernize-use-std-print)
            "%s\n",
            std::format(
                "The Lernaean hydra has {} heads. They all spoke at once, inside the graph, and said: '{}'.",
                pattern.size(),
                got)
                .c_str());

        if (got != pattern)
            throw std::runtime_error(
                std::format(
                    "Heracles got to a head: got '{}', expected '{}'. Fetch Iolaus and a torch.", got, pattern));

        std::printf( // NOLINT(modernize-use-std-print)
            "Every head survived the labours. Want a bigger monster? Add characters to 'pattern'.\n");
    }

    void run() const {
        const Kokkos::Execution::GraphContext<Exec> gctx{this->exec};

        auto sndr = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return stdexec::when_all(
                (stdexec::schedule(gctx.get_scheduler())
                 | stdexec::then(Write<view_t, pattern.at(Is), Is>{.data = data}))...);
        }(std::make_index_sequence<pattern.size()>{});

        stdexec::sync_wait(std::move(sndr));

        check();
    }

   protected:
    view_t data;
};

} // namespace Examples::Graph

int main(int argc, char* argv[]) {
    const Kokkos::ScopeGuard guard{argc, argv};
    {
        Examples::Graph::Hydra<EXAMPLE_EXECUTION_SPACE>{}.run();
    }
    return EXIT_SUCCESS;
}
