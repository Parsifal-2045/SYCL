#include <CL/sycl.hpp>

#ifndef INTEL_GPU_SELECTOR
#define INTEL_GPU_SELECTOR

class Intel_gpu_selector : public sycl::device_selector
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

#endif