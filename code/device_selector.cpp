#include <CL/sycl.hpp>

class intel_gpu_selector : public sycl::device_selector
{
    int operator() (const sycl::device &dev) const override
    {
        if (dev.has(sycl::aspect::gpu))
        {
            auto vendorName = dev.get_info<sycl::info::device::vendor>();
            if(vendorName.find("Intel") != std::string::npos)
            {
                return 1;
            }
        }
        return -1;
    }
};

int main()
{
    auto queue = sycl::queue{intel_gpu_selector{}};
    std::cout << "Chosen device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';
}

// Returns -1 stating "No device of requested type is available"