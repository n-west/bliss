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
std::pair<std::map<int, config::cuda_device_attributes>, std::vector<int>> check_cuda_architectures() {
#if BLAND_CUDA_CODE
    int device_count;
    auto cu_result = cudaGetDeviceCount(&device_count);

    if (cu_result != cudaSuccess) {
        fmt::print("ERROR: CUDA error {} when trying to get device count\n", static_cast<int>(cu_result));
        if (cu_result == cudaErrorCompatNotSupportedOnDevice) {
            fmt::print("ERROR: Got cuda error cudaErrorCompatNotSupportedOnDevice which usually means a driver/library"
            " version mismatch. If the nvidia driver was recently upgraded try reloading it. sudo rmmod nvidia_drm;"
            " sudo rmmod nvidia_modeset; sudo rmmod nvidia_uvm; sudo rmmod nvidia\n");
        }
        return {{},{}};
    }

    if (device_count == 0) {
        fmt::print("WARN: no CUDA devices found. Will only use CPU\n");
        return {{},{}};
    }

    std::map<int, config::cuda_device_attributes> useable_devices;
    std::vector<int> nonuseable_devices;

    cudaDeviceProp deviceProp;
    for (int device_index = 0; device_index < device_count; ++device_index) {
        cudaGetDeviceProperties(&deviceProp, device_index);

        int runtime_arch = deviceProp.major * 10 + deviceProp.minor;

        if (std::find(compiled_archs.begin(), compiled_archs.end(), runtime_arch) != compiled_archs.end()) {
            auto uuid_bytes = deviceProp.uuid.bytes;
            useable_devices[device_index] = {.name=deviceProp.name, .uuid={}};
            for (int ii=0; ii < sizeof(deviceProp.uuid); ++ii) {
                useable_devices[device_index].uuid += fmt::format("{:02x}", uuid_bytes[ii]);
            }
            fmt::print("Found a usable device\n");
        } else {
            fmt::print("WARN: Device {}/{} has compute architecture {}.{} which is incompatible with this build.\n", device_index, device_count, deviceProp.major, deviceProp.minor);
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


std::map<int, config::cuda_device_attributes> config::get_valid_cuda_devices() {
    return _valid_cuda_devices;
}


bool config::check_is_valid_cuda_device(int device_index, bool verbose) {
    bool valid_device = std::any_of(_valid_cuda_devices.begin(), _valid_cuda_devices.end(), [device_index](std::pair<int, cuda_device_attributes> valid_dev) {return valid_dev.first == device_index;});
    if (verbose) {
        if (valid_device) {
            fmt::print("INFO: using cuda:{} :: {} (UUID: {})\n", device_index, _valid_cuda_devices[device_index].name, _valid_cuda_devices[device_index].uuid);
        } else {
            fmt::print("The selected device id either does not exist or has a compute capability that is not compatible with this build\n");
        }
    }
    return valid_device;
}

}

