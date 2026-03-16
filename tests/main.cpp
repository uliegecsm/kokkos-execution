#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

//! Entry point that will initialize both <tt>Google Test</tt> and @c Kokkos.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //! Instruct the tools to avoid global fences if possible.
    Kokkos::Tools::Experimental::set_request_tool_settings_callback(
        [](const uint32_t, Kokkos::Tools::Experimental::ToolSettings* settings) -> void {
            settings->requires_global_fencing = false;
        });

    Kokkos::initialize(argc, argv);

    Kokkos::print_configuration(std::cout);

    const auto code = RUN_ALL_TESTS();

    Kokkos::finalize();

    return code;
}
