#include <core/frequency_drift_plane.hpp>


using namespace bliss;


bliss::frequency_drift_plane::frequency_drift_plane(bland::ndarray drift_plane, integrated_flags drift_rfi) : _integrated_drifts(drift_plane), _dedrifted_rfi(drift_rfi), _device(drift_plane.device()) {}

bliss::frequency_drift_plane::frequency_drift_plane(bland::ndarray drift_plane, integrated_flags drift_rfi, int64_t integration_steps, std::vector<bliss::frequency_drift_plane::drift_rate> dri) : 
    _integrated_drifts(drift_plane), _dedrifted_rfi(drift_rfi), _integration_steps(integration_steps), _drift_rate_info(dri), _device(drift_plane.device()) {
}

int64_t bliss::frequency_drift_plane::integration_steps() {
    return _integration_steps;
}

std::vector<bliss::frequency_drift_plane::drift_rate> bliss::frequency_drift_plane::drift_rate_info() {
    return _drift_rate_info;
}

bland::ndarray bliss::frequency_drift_plane::integrated_drift_plane(){
    return _integrated_drifts;
}

integrated_flags bliss::frequency_drift_plane::integrated_rfi(){
    return _dedrifted_rfi;
}

void bliss::frequency_drift_plane::set_device(bland::ndarray::dev dev) {
    _device = dev;
    _dedrifted_rfi.set_device(_device);
    _integrated_drifts = _integrated_drifts.to(_device);
}

void bliss::frequency_drift_plane::set_device(std::string_view dev_str) {
    auto dev = bland::ndarray::dev(dev_str);
    set_device(dev);
}

