
#include <drift_search/connected_components.hpp>

#include "kernels/connected_components_cpu.hpp"
#if BLISS_CUDA
#include "kernels/connected_components_cuda.cuh"
#endif // BLISS_CUDA

#include <fmt/format.h>

#include <stdexcept>

using namespace bliss;

std::vector<protohit> bliss::find_components_in_binary_mask(const bland::ndarray         &mask,
                                                            std::vector<bland::nd_coords> neighborhood) {
    return find_components_in_binary_mask_cpu(mask, neighborhood);
}

std::vector<protohit> bliss::find_components_above_threshold(bland::ndarray                   doppler_spectrum,
                                                             integrated_flags                 dedrifted_rfi,
                                                             float                            noise_floor,
                                                             std::vector<protohit_drift_info> noise_per_drift,
                                                             float                            snr_threshold,
                                                             std::vector<bland::nd_coords>    max_neighborhood) {

    auto compute_device = doppler_spectrum.device();
#if BLISS_CUDA
    if (compute_device.device_type == bland::ndarray::dev::cuda.device_type) {
        return find_components_above_threshold_cuda(doppler_spectrum, dedrifted_rfi, noise_floor, noise_per_drift, snr_threshold, max_neighborhood);
    } else
#endif // BLISS_CUDA
    if (compute_device.device_type == bland::ndarray::dev::cpu.device_type) {
        return find_components_above_threshold_cpu(doppler_spectrum, dedrifted_rfi, noise_floor, noise_per_drift, snr_threshold, max_neighborhood);
    } else {
        throw std::runtime_error("Unsupported device for find_components_above_threshold");
    }
}
