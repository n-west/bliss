#pragma once

#if BLAND_CUDA_CODE
#include <cuda_runtime.h>
#endif

#include <map>
#include <string>
#include <vector>

namespace bland {

/**
 * a config singleton that allows start-up/init-time based discoverability
 * of devices and setting global parameters
*/
class config {
public:

// #if BLAND_CUDA_CODE
//     using cuda_device_attributes = cudaDeviceProp;
// #else
    // Here to make the conditional compilation of used fields restricted to one place
    struct cuda_device_attributes {
        std::string name;                  /**< ASCII string identifying device */
        std::string uuid;                   /**< 16-byte unique identifier */
    };
// #endif

    static config& get_instance() {
        static config instance;
        return instance;
    }

    config(config const&) = delete;
    void operator=(config const&) = delete;

    std::map<int, cuda_device_attributes> get_valid_cuda_devices();
    bool check_is_valid_cuda_device(int device_index, bool verbose=false);

private:
    config();

    std::map<int, cuda_device_attributes> _valid_cuda_devices;
    std::vector<int> _invalid_cuda_devices;
    
};

static auto& g_config = config::get_instance();


} // namespace bland