#include "local_maxima_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


__global__ void local_maxima_kernel(float* doppler_spectrum_data,
    protohit_drift_info* noise_per_drift,
    int32_t number_drifts, int32_t number_channels,
    float noise_floor,
    float snr_threshold,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits) {

    // Strategy:
    // Each thread will compare a candidate local_max (every point in grid) to its neighborhood

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    auto numel = number_drifts * number_channels;
    for (int n=tid; n < numel; n+=grid_size) {
        int central_frequency_channel = n % number_channels;
        int central_drift_index = n / number_channels;

        int central_index = central_drift_index * number_channels + central_frequency_channel;

        auto candidate_maxima_val = doppler_spectrum_data[central_index];
        auto noise_at_this_drift = noise_per_drift[central_drift_index].integration_adjusted_noise;
        auto candidate_snr = (candidate_maxima_val - noise_floor) / noise_at_this_drift;
        if (candidate_snr > snr_threshold) {
            // 4. Check if it is greater than surrounding neighborhood
            bool neighborhood_max = true;
            for (int freq_neighbor_offset = -neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                for (int drift_neighbor_offset = -neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist - abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                    auto neighbor_coord = freq_drift_coord{.drift_index=central_drift_index, .frequency_channel=central_frequency_channel};

                    neighbor_coord.drift_index += drift_neighbor_offset;
                    neighbor_coord.frequency_channel += freq_neighbor_offset;

                    if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                        neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                        auto linear_neighbor_index =
                                neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;
                        auto neighbor_val   = doppler_spectrum_data[linear_neighbor_index];
                        auto neighbor_noise = noise_per_drift[neighbor_coord.drift_index].integration_adjusted_noise;
                        auto neighbor_snr   = (neighbor_val - noise_floor) / neighbor_noise;
                        if (neighbor_snr > candidate_snr) {
                            neighborhood_max = false;
                            goto neighborhood_checked;
                        }
                    } else {
                        neighborhood_max = false;
                        goto neighborhood_checked;
                    }
                }
            }
            neighborhood_checked:
            if (neighborhood_max) {
                auto protohit_index = atomicAdd(number_protohits, 1);
                protohits[protohit_index].index_max = {.drift_index=central_drift_index, .frequency_channel=central_frequency_channel};
                protohits[protohit_index].snr = candidate_snr;
                protohits[protohit_index].max_integration = candidate_maxima_val;
                protohits[protohit_index].desmeared_noise = noise_at_this_drift;
                // TODO: figure out how to get rfi counts in

                // At the local max drift rate, look up and down in frequency dim adding locations
                // that are above the SNR threshold AND continuing to decrease
                auto max_lower_channel = central_frequency_channel;
                auto lower_edge_value = candidate_maxima_val;
                bool expand_band_down = true;
                do {
                    max_lower_channel -= 1;

                    if (max_lower_channel >= 0) {
                        int linear_expansion_index = central_drift_index * number_channels + max_lower_channel;

                        auto new_lower_edge_value = doppler_spectrum_data[linear_expansion_index];
                        auto new_lower_edge_snr = (new_lower_edge_value - noise_floor) / noise_at_this_drift;
                        if (new_lower_edge_snr > snr_threshold && new_lower_edge_value < lower_edge_value) {
                            lower_edge_value = new_lower_edge_value;
                            // push back locations
                        } else {
                            expand_band_down = false;
                        }
                    } else {
                        expand_band_down = false;
                    }

                } while (expand_band_down);


                auto max_upper_channel = central_frequency_channel;
                auto upper_edge_value = candidate_maxima_val;
                bool expand_band_up = true;
                do {
                    max_upper_channel += 1;

                    if (max_upper_channel < number_channels) {
                        int linear_expansion_index = central_drift_index * number_channels + max_upper_channel;

                        auto new_upper_edge_value = doppler_spectrum_data[linear_expansion_index];
                        auto new_upper_edge_snr = (new_upper_edge_value - noise_floor) / noise_at_this_drift;
                        if (new_upper_edge_snr > snr_threshold && new_upper_edge_value < upper_edge_value) {
                            upper_edge_value = new_upper_edge_value;
                            // push back locations
                        } else {
                            expand_band_up = false;
                        }
                    } else {
                        expand_band_up = false;
                    }

                } while (expand_band_up);

                auto binwidth = max_upper_channel - max_lower_channel;
                protohits[protohit_index].binwidth = binwidth;
                protohits[protohit_index].index_center = {.drift_index=central_drift_index, .frequency_channel=(max_upper_channel + max_lower_channel)/2};
            }
        }
    }
}

std::vector<protohit>
bliss::find_local_maxima_above_threshold_cuda(bland::ndarray                 doppler_spectrum,
                                            integrated_flags                 dedrifted_rfi,
                                            float                            noise_floor,
                                            std::vector<protohit_drift_info> noise_per_drift,
                                            float                            snr_threshold,
                                            int                              neighbor_l1_dist) {

    auto doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto doppler_spectrum_strides = doppler_spectrum.strides();
    auto doppler_spectrum_shape  = doppler_spectrum.shape();

    int32_t number_drifts = doppler_spectrum_shape[0];
    int32_t number_channels = doppler_spectrum_shape[1];

    auto numel = doppler_spectrum.numel();

    thrust::device_vector<protohit_drift_info> dev_noise_per_drift(noise_per_drift.begin(), noise_per_drift.end());

    // We can only possibly have one max for every neighborhood, so that's a reasonably efficient max neighborhood
    thrust::device_vector<device_protohit> dev_protohits(numel / (neighbor_l1_dist*neighbor_l1_dist));

    int* dev_num_maxima;
    cudaMallocManaged(&dev_num_maxima, sizeof(int));
    *dev_num_maxima = 0;

    int num_maxima = 0;
    cudaMemcpy(dev_num_maxima, &num_maxima, sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 512;
    int number_blocks = 1;
    int smem = 0;
    local_maxima_kernel<<<block_size, number_blocks, smem>>>(
        doppler_spectrum_data,
        thrust::raw_pointer_cast(dev_noise_per_drift.data()),
        number_drifts, number_channels,
        noise_floor, snr_threshold,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        dev_num_maxima
    );

    cudaDeviceSynchronize();

    dev_protohits.resize(*dev_num_maxima);
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
