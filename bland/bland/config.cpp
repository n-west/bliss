#include "bland/config.hpp"

#include "generated/bland_cuda_archs.hpp"

#if BLAND_CUDA_CODE
#include <cuda_runtime.h>
#endif
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>

namespace bland {

// list of good and bad (useable & not useable) devices by device id
std::pair<std::vector<int>, std::vector<int>> check_cuda_architectures() {
#if BLAND_CUDA_CODE
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        fmt::print("WARN: no CUDA devices found. Will only use CPU\n");
        return {{},{}};
    }

    std::vector<int> useable_devices;
    std::vector<int> nonuseable_devices;

    for (int device_index = 0; device_index < device_count; ++device_index) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_index);

        int runtime_arch = deviceProp.major * 10 + deviceProp.minor;

        if (std::find(compiled_archs.begin(), compiled_archs.end(), runtime_arch) != compiled_archs.end()) {
            useable_devices.emplace_back(device_index);
        } else {
            fmt::print("WARN: Device {} has compute architecture {}.{} which is incompatible with this build.\n", device_index, deviceProp.major, deviceProp.minor);
            nonuseable_devices.emplace_back(device_index);
        }
    }
    return {useable_devices, nonuseable_devices};
#else
    // no cuda code, so no devices
    return {{},{}};
#endif
}

    config::config()  {
        auto devices = check_cuda_architectures();
        _valid_cuda_devices = devices.first;
        _invalid_cuda_devices = devices.second;
    }


    std::vector<int> config::get_valid_cuda_devices() {
        return _valid_cuda_devices;
    }

    bool config::check_is_valid_cuda_device(int device_index) {
        return std::any_of(_valid_cuda_devices.begin(), _valid_cuda_devices.end(), [device_index](int valid_dev) {return valid_dev == device_index;});
    }

}

