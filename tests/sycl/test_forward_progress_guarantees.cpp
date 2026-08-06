#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#if !defined(SYCL_EXT_ONEAPI_FORWARD_PROGRESS)
#    error "This test requires the 'sycl_ext_oneapi_forward_progress' extension."
#endif

/**
 * Forward progress guarantees from the @c SYCL API
 * ------------------------------------------------
 *
 * The 'sycl_ext_oneapi_forward_progress' is described in
 * https://github.com/intel/llvm/blob/86fb52b022673676ec3216a7ae344ea0fbd91790/sycl/doc/extensions/experimental/sycl_ext_oneapi_forward_progress.asciidoc,
 * and stems from @cite pennycook-alignment-sycl-parallelism-with-cpp.
 */

namespace Tests::SYCL {

//! Stream operator for @c sycl::ext::oneapi::experimental::forward_progress_guarantee.
std::ostream& operator<<(std::ostream& out, const sycl::ext::oneapi::experimental::forward_progress_guarantee& fpg) {
    switch (fpg) {
    case sycl::ext::oneapi::experimental::forward_progress_guarantee::concurrent:
        return out << "concurrent";
    case sycl::ext::oneapi::experimental::forward_progress_guarantee::parallel:
        return out << "parallel";
    case sycl::ext::oneapi::experimental::forward_progress_guarantee::weakly_parallel:
        return out << "weakly_parallel";
    default:
        std::abort();
    }
}

//! Stream operator for a @c sycl::device.
std::ostream& operator<<(std::ostream& out, const sycl::device& device) {
    out << "Device:\n";
    out << "\t- " << device.get_info<sycl::info::device::name>() << '\n';
    out << "\t- " << device.get_info<sycl::info::device::version>() << '\n';
    out << "\t- " << device.get_info<sycl::info::device::driver_version>() << '\n';
    out << "\t- " << device.get_info<sycl::info::device::vendor>() << '\n';
    return out;
}

struct SYCLForwardProgressGuaranteesTest : public testing::Test {
   public:
    SYCLForwardProgressGuaranteesTest()
        : device(sycl::queue{}.get_device()) {
        std::cout << device << std::endl;
    }
   protected:
    sycl::device device;
};

/**
 * @test Check the progress capabilities queries for the device.
 *
 * Inspired by https://github.com/intel/llvm/blob/be755dfff931b6eed0d344114a9812c924dd49fd/sycl/test-e2e/forward_progress/forward_progress_L0_gpu.cpp#L6.
 */
TEST_F(SYCLForwardProgressGuaranteesTest, progress_capabilities) {
    using namespace sycl::ext::oneapi::experimental;

    {
        const auto pc_wg_rg =
            device.get_info<info::device::work_group_progress_capabilities<execution_scope::root_group>>();
        ASSERT_THAT(pc_wg_rg, testing::SizeIs(1));
        std::cout << "Work-group at the root-group scope: " << pc_wg_rg.at(0) << std::endl;
    }
    {
        const auto pc_sg_rg =
            device.get_info<info::device::sub_group_progress_capabilities<execution_scope::root_group>>();
        ASSERT_THAT(pc_sg_rg, testing::SizeIs(1));
        std::cout << "Sub-group at the root-group scope: " << pc_sg_rg.at(0) << std::endl;
    }
    {
        const auto pc_wi_rg =
            device.get_info<info::device::work_item_progress_capabilities<execution_scope::root_group>>();
        ASSERT_THAT(pc_wi_rg, testing::SizeIs(1));
        std::cout << "Work-item at the root-group scope: " << pc_wi_rg.at(0) << std::endl;
    }

    {
        const auto pc_sg_wg =
            device.get_info<info::device::sub_group_progress_capabilities<execution_scope::work_group>>();
        ASSERT_THAT(pc_sg_wg, testing::SizeIs(1));
        std::cout << "Sub-group at the work-group scope: " << pc_sg_wg.at(0) << std::endl;
    }
    {
        const auto pc_wi_wg =
            device.get_info<info::device::work_item_progress_capabilities<execution_scope::work_group>>();
        ASSERT_THAT(pc_wi_wg, testing::SizeIs(1));
        std::cout << "Work-item at the work-group scope: " << pc_wi_wg.at(0) << std::endl;
    }

    {
        const auto pc_wi_sg =
            device.get_info<info::device::work_item_progress_capabilities<execution_scope::sub_group>>();
        ASSERT_THAT(pc_wi_sg, testing::SizeIs(1));
        std::cout << "Work-item at the sub-group scope: " << pc_wi_sg.at(0) << std::endl;
    }
}

template <typename T>
struct Functor {
    Functor(const T& props_)
        : props(props_) {
    }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return props;
    }

    KOKKOS_FUNCTION
    void operator()() const {
    }

    T props;
};

/**
 * @test Check that kernels properties requesting certain forward progress guarantees are honored.
 *
 * Inspired by https://github.com/intel/llvm/blob/be755dfff931b6eed0d344114a9812c924dd49fd/sycl/test-e2e/forward_progress/forward_progress_kernel_param_L0_gpu.cpp.
 */
TEST_F(SYCLForwardProgressGuaranteesTest, exceeds_the_scheduler_guarantees) {
    using namespace sycl::ext::oneapi::experimental;

    const auto pc_wi_rg = device.get_info<info::device::work_item_progress_capabilities<execution_scope::sub_group>>();
    ASSERT_THAT(pc_wi_rg, testing::SizeIs(1));
    ASSERT_EQ(pc_wi_rg.at(0), forward_progress_guarantee::weakly_parallel);

    sycl::queue q{device};

    //! Requesting weakly-parallel is fine, the device supports it.
    q.single_task(
        Functor{
            properties{work_group_progress<forward_progress_guarantee::weakly_parallel, execution_scope::root_group>}});

    //! Requesting forward progress guarantees that the device does not support throws.
    ASSERT_THAT(
        [&]() {
            q.single_task(
                Functor{properties{
                    work_group_progress<forward_progress_guarantee::parallel, execution_scope::root_group>}});
        },
        ::testing::ThrowsMessage<sycl::exception>(
            ::testing::HasSubstr("Required progress guarantee for work groups is not supported by this device.")));
    ASSERT_THAT(
        [&]() {
            q.single_task(
                Functor{properties{
                    work_group_progress<forward_progress_guarantee::concurrent, execution_scope::root_group>}});
        },
        ::testing::ThrowsMessage<sycl::exception>(
            ::testing::HasSubstr("Required progress guarantee for work groups is not supported by this device.")));
}

} // namespace Tests::SYCL
