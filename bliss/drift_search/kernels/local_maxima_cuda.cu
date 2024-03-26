#include "local_maxima_cuda.cuh"

using namespace bliss;

__global__ void local_maxima_kernel() {

}

std::vector<protohit>
bliss::find_local_maxima_above_threshold_cuda(bland::ndarray                      doppler_spectrum,
                                             integrated_flags                     dedrifted_rfi,
                                             std::vector<std::pair<float, float>> noise_and_thresholds_per_drift,
                                             std::vector<bland::nd_coords>        max_neighborhood) {


    int block_size = 512;
    int number_blocks = 1;
    int smem = 0;
    local_maxima_kernel<<<block_size, number_blocks, smem>>>();

    return {};
}
