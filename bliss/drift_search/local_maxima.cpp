
#include <drift_search/local_maxima.hpp>

#include <bland/ndarray.hpp>

#include "kernels/local_maxima_cpu.hpp"
#if BLISS_CUDA
#include "kernels/local_maxima_cuda.cuh"
#endif // BLISS_CUDA

#include <fmt/format.h>

using namespace bliss;


std::vector<protohit> bliss::find_local_maxima_above_threshold(bland::ndarray doppler_spectrum,
                                        integrated_flags                      dedrifted_rfi,
                                        float                                 noise_floor,
                                        std::vector<protohit_drift_info>      noise_per_drift,
                                        float                                 snr_threshold,
                                        int                                   neighbor_l1_dist) {

    auto compute_device = doppler_spectrum.device();
#if BLISS_CUDA
    if (compute_device.device_type == bland::ndarray::dev::cuda.device_type) {
        return find_local_maxima_above_threshold_cuda(doppler_spectrum, dedrifted_rfi, noise_floor, noise_per_drift, snr_threshold, neighbor_l1_dist);
    } else 
#endif // BLISS_CUDA
    if (compute_device.device_type == bland::ndarray::dev::cpu.device_type) {
        return find_local_maxima_above_threshold_cpu(doppler_spectrum, dedrifted_rfi, noise_floor, noise_per_drift, snr_threshold, neighbor_l1_dist);
    } else {
        throw std::runtime_error("Unsupported device for find_local_maxima_above_threshold");
    }
}
