#include "connected_components_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


__global__ void find_components_kernel(float* doppler_spectrum_data,
    protohit_drift_info* noise_per_drift,
    int32_t number_drifts, int32_t number_channels,
    float noise_floor,
    float snr_threshold,
    freq_drift_coord* neighbor_offsets,
    int neighborhood_size,
    device_protohit* protohits,
    int* number_protohits) {

    // Strategy:
    // Each thread will compare a candidate local_max (every point in grid) to its neighborhood
    // Make space to eventually allow (small) thread blocks to do a parallel neighborwide search
    // so work unit is by thread block instead of per thread

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;
    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto candidate_val = doppler_spectrum_data[candidate_linear_index];
        auto noise_at_candidate_drift = noise_per_drift[candidate_drift_index].integration_adjusted_noise;
        auto candidate_snr = candidate_val / noise_at_candidate_drift;
        if (candidate_snr > snr_threshold) {
            // Keep expanding the protohit from here. If connected to a higher SNR, abort
            // If this is the highest SNR, add it as a protohit
            bool protohit_is_valid = true;

            // Push this one to a queue

            // Iterate over the queue until it stops expanding
        }
    }

}

std::vector<protohit>
bliss::find_components_above_threshold_cuda(bland::ndarray                 doppler_spectrum,
                                            integrated_flags                 dedrifted_rfi,
                                            float                            noise_floor,
                                            std::vector<protohit_drift_info> noise_per_drift,
                                            float                            snr_threshold,
                                            std::vector<bland::nd_coords>    max_neighborhood) {

    auto doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto doppler_spectrum_strides = doppler_spectrum.strides();
    auto doppler_spectrum_shape  = doppler_spectrum.shape();

    int32_t number_drifts = doppler_spectrum_shape[0];
    int32_t number_channels = doppler_spectrum_shape[1];

    auto numel = doppler_spectrum.numel();

    thrust::device_vector<protohit_drift_info> dev_noise_per_drift(noise_per_drift.begin(), noise_per_drift.end());

    // This assumes that every array has the same shape/strides (which should be true!!! but needs to be checked)
    std::vector<freq_drift_coord> neighborhood_offsets;
    neighborhood_offsets.reserve(max_neighborhood.size());
    for (auto neighbor_coords : max_neighborhood) {
        auto n = freq_drift_coord{.drift_index=static_cast<int32_t>(neighbor_coords[0]), .frequency_channel=static_cast<int32_t>(neighbor_coords[1])};
        neighborhood_offsets.emplace_back(n);
    }
    thrust::device_vector<freq_drift_coord> device_neighborhood(neighborhood_offsets.begin(), neighborhood_offsets.end());
    // We can only possibly have one max for every neighborhood, so that's a reasonably efficient max neighborhood
    thrust::device_vector<device_protohit> dev_protohits(numel / max_neighborhood.size());

    int* dev_num_maxima;
    cudaMalloc((void**)&dev_num_maxima, sizeof(int));

    int num_maxima = 0;
    cudaMemcpy(dev_num_maxima, &num_maxima, sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 512;
    int number_blocks = 1;
    int smem = 0;
    find_components_kernel<<<block_size, number_blocks, smem>>>(
        doppler_spectrum_data,
        thrust::raw_pointer_cast(dev_noise_per_drift.data()),
        number_drifts, number_channels,
        noise_floor, snr_threshold,
        thrust::raw_pointer_cast(device_neighborhood.data()),
        device_neighborhood.size(),
        thrust::raw_pointer_cast(dev_protohits.data()),
        dev_num_maxima
    );

    cudaMemcpy(&num_maxima, dev_num_maxima, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_num_maxima);

    dev_protohits.resize(num_maxima);
    thrust::host_vector<device_protohit> host_protohits(dev_protohits.begin(), dev_protohits.end());

    std::vector<protohit> export_protohits;
    export_protohits.reserve(host_protohits.size());
    for (auto &simple_hit : host_protohits) {
        auto new_protohit = protohit();
        new_protohit.index_max = simple_hit.index_max;
        new_protohit.index_center = simple_hit.index_center;
        new_protohit.snr = simple_hit.snr;
        new_protohit.max_integration = simple_hit.max_integration;
        new_protohit.desmeared_noise = simple_hit.desmeared_noise;
        new_protohit.binwidth = simple_hit.binwidth;

        // new_protohit.rfi_counts = simple_hit.desmeared_noise;

        export_protohits.push_back(new_protohit);
    }

    return export_protohits;
}
