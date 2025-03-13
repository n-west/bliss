#include <core/frequency_drift_plane.hpp>


using namespace bliss;


bliss::frequency_drift_plane::frequency_drift_plane(bland::ndarray drift_plane, integrated_flags drift_rfi) : _integrated_drifts(drift_plane), _dedrifted_rfi(drift_rfi), _device(drift_plane.device()) {}

bliss::frequency_drift_plane::frequency_drift_plane(bland::ndarray drift_plane, integrated_flags drift_rfi, std::vector<bliss::frequency_drift_plane::drift_rate> dri) : 
    _integrated_drifts(drift_plane), _dedrifted_rfi(drift_rfi), _drift_rate_info(dri), _device(drift_plane.device()) {
}

std::vector<bliss::frequency_drift_plane::drift_rate> bliss::frequency_drift_plane::drift_rate_info() {
    return _drift_rate_info;
}

bland::ndarray bliss::frequency_drift_plane::integrated_drift_plane() {
    _integrated_drifts = _integrated_drifts.to(_device);
    return _integrated_drifts;
}

integrated_flags bliss::frequency_drift_plane::integrated_rfi() {
    _dedrifted_rfi.set_device(_device);
    _dedrifted_rfi.push_device();
    return _dedrifted_rfi;
}

void bliss::frequency_drift_plane::set_device(bland::ndarray::dev dev) {
    _device = dev;
}

void bliss::frequency_drift_plane::set_device(std::string_view dev_str) {
    auto dev = bland::ndarray::dev(dev_str);
    set_device(dev);
}

void bliss::frequency_drift_plane::push_device() {
    _dedrifted_rfi.set_device(_device);
    _dedrifted_rfi.push_device();
    _integrated_drifts = _integrated_drifts.to(_device);
}


