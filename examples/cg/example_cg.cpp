#include <iostream>

#include "Kokkos_Core.hpp"

namespace Examples {

class CG {
   public:
    static void run(const EXAMPLE_EXECUTION_SPACE&) {
        std::cout << "Hello!" << std::endl;
    }
};

} // namespace Examples

int main(int argc, char* argv[]) {
    const Kokkos::ScopeGuard guard{argc, argv};
    {
        Examples::CG::run(EXAMPLE_EXECUTION_SPACE{});
    }
}
