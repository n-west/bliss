
#include <drift_search/local_maxima.hpp>

#include "kernels/local_maxima_cpu.hpp"
#include "kernels/local_maxima_cuda.cuh"

using namespace bliss;


std::vector<protohit> bliss::find_local_maxima_above_threshold(bland::ndarray doppler_spectrum,
                                        integrated_flags                      dedrifted_rfi,
                                        std::vector<std::pair<float, float>>  noise_and_thresholds_per_drift,
                                        std::vector<bland::nd_coords>         max_neighborhood) {

    return find_local_maxima_above_threshold_cpu(doppler_spectrum, dedrifted_rfi, noise_and_thresholds_per_drift, max_neighborhood);
}
