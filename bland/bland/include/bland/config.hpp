#pragma once

#include <vector>

namespace bland {

/**
 * a config singleton that allows start-up/init-time based discoverability
 * of devices and setting global parameters
*/
class config {
public:
    static config& get_instance() {
        static config instance;
        return instance;
    }

    config(config const&) = delete;
    void operator=(config const&) = delete;

    std::vector<int> get_valid_cuda_devices();
    bool check_is_valid_cuda_device(int device_index);

private:
    config();

    std::vector<int> _valid_cuda_devices;
    std::vector<int> _invalid_cuda_devices;
    
};

static auto& g_config = config::get_instance();


} // namespace bland