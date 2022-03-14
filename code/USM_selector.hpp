#include <CL/sycl.hpp>

#ifndef USM_SELECTOR
#define USM_SELECTOR

class USM_device_selector : public sycl::device_selector
{
    int operator()(const sycl::device &dev) const override
    {
        if (dev.has(sycl::aspect::usm_device_allocations))
        {
            return 1;
        }
        return -1;
    }
};

// returns Intel FPGA Emulation Device

#endif