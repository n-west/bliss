#include "connected_components_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


__global__ void find_components_kernel(float* doppler_spectrum_data,
    uint32_t *visited,
    protohit_drift_info* noise_per_drift,
    int32_t number_drifts, int32_t number_channels,
    float noise_floor,
    float snr_threshold,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits) {

    // Strategy:
    // 1) grid-strided loop to label each bin in freq-drift plane as a local_maxima with a unique id, above SNR threshold,
    // above SNR/2 threshold, or not above SNR threshold
    // 2) while changed > 0:
    //   A) grid-stride loop and replace the neighborhood with the highest index of the neighborhood

    // visited possible values:
    // 0: not visited
    // 1: visited, not above SNR threshold
    // 2: visited, above SNR/2
    // 3: visited, above SNR
    // >=8: visited, above SNR and local_maxima

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;
    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        if (visited[candidate_linear_index] == 0) {
            visited[candidate_linear_index] = 1;
            auto candidate_val = doppler_spectrum_data[candidate_linear_index];
            auto noise_at_candidate_drift = noise_per_drift[candidate_drift_index].integration_adjusted_noise;
            auto candidate_snr = (candidate_val - noise_floor) / noise_at_candidate_drift;
            if (candidate_snr > snr_threshold) {
                visited[candidate_linear_index] = 3;

                // Set to a unique id if this is a local maxima
                bool neighborhood_max = true;
                for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                    for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                        auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};

                        neighbor_coord.drift_index += drift_neighbor_offset;
                        neighbor_coord.frequency_channel += freq_neighbor_offset;

                        if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                                neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                            auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;
                            auto neighbor_val = doppler_spectrum_data[linear_neighbor_index];
                            auto neighbor_noise = noise_per_drift[neighbor_coord.drift_index].integration_adjusted_noise;
                            auto neighbor_snr = (neighbor_val - noise_floor) / neighbor_noise;
                            if (neighbor_snr > candidate_snr) {
                                neighborhood_max = false;
                                break; // break sounds right, but may lead to warp divergeance. Benchmark!
                            }
                        } else {
                            neighborhood_max = false;
                            break;
                        }
                    }
                }
                if (neighborhood_max) {
                    auto protohit_index = atomicAdd(number_protohits, 1);
                    protohits[protohit_index].index_max = {.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
                    protohits[protohit_index].snr = candidate_snr;
                    protohits[protohit_index].max_integration = candidate_val;
                    protohits[protohit_index].desmeared_noise = noise_at_candidate_drift;

                    visited[candidate_linear_index] = protohit_index;

                    // Assign visited = protohit_index for every neighbor
                    for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                        for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                            auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};

                            neighbor_coord.drift_index += drift_neighbor_offset;
                            neighbor_coord.frequency_channel += freq_neighbor_offset;

                                if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                                        neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                                    auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;
                                    auto neighbor_val = doppler_spectrum_data[linear_neighbor_index];
                                    auto neighbor_noise = noise_per_drift[neighbor_coord.drift_index].integration_adjusted_noise;
                                    auto neighbor_snr = (neighbor_val - noise_floor) / neighbor_noise;
                                    if (neighbor_snr > candidate_snr/2) {
                                        visited[linear_neighbor_index] = protohit_index;
                                    }
                                }
                        }
                    }
                }

            } else if (candidate_snr > snr_threshold/2) {
                visited[candidate_linear_index] = 2;
            }
        } else {

        }
    }
}

__device__ int find_neighbor_max_snr_protohit_id(uint32_t *visited,
    int number_drifts, int number_channels,
    freq_drift_coord central_point,
    int candidate_highest_snr_protohit_id,
    device_protohit* protohits,
    freq_drift_coord* neighbor_offsets,
    int neighborhood_size
) {
    double highest_snr = protohits[candidate_highest_snr_protohit_id].snr;

    for (int neighbor_index=0; neighbor_index < neighborhood_size; ++neighbor_index) {
        auto neighbor_offset = neighbor_offsets[neighbor_index];
        auto neighbor_coord = central_point;

        neighbor_coord.drift_index += neighbor_offset.drift_index;
        neighbor_coord.frequency_channel += neighbor_offset.frequency_channel;

        if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

            auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;

            auto neighbor_id = visited[linear_neighbor_index];
        }
    }
}

__global__ void spread_components_kernel(
    uint32_t *visited,
    int32_t number_drifts, int32_t number_channels,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits,
    int* invalidated_protohits,
    int* changes) {

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;

    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto protohit_id = visited[candidate_linear_index];
        
        if (protohit_id >= 8 && !protohits[protohit_id].valid) {
            visited[candidate_linear_index] = protohits[protohit_id].invalidated_by;
            protohit_id = protohits[protohit_id].invalidated_by;
        }
        if (protohit_id >= 2) {
            auto protohit = protohits[protohit_id];

            double neighborhood_max_snr = protohit.snr;
            int neighborhood_max_id = -1;
            // bool neighborhood_max = true;
            // Expand the neighborhood, adopting the highest SNR protohit within the neighborhood if needed
            for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                    auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};

                    neighbor_coord.drift_index += drift_neighbor_offset;
                    neighbor_coord.frequency_channel += freq_neighbor_offset;

                    if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                            neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                        auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;

                        auto neighbor_id = visited[linear_neighbor_index];
                        if (neighbor_id >= 8) {
                            auto neighbor_protohit = protohits[neighbor_id];
                            if (neighbor_protohit.snr > neighborhood_max_snr) {
                                protohits[protohit_id].index_max = {.drift_index=neighbor_coord.drift_index, .frequency_channel=neighbor_coord.frequency_channel};
                                neighborhood_max_snr = protohits[neighbor_id].snr;
                                neighborhood_max_id = neighbor_id;
                            }
                        }
                    }
                }
            }
            if (neighborhood_max_id > 0) {
                if (protohits[protohit_id].valid) {
                    protohits[protohit_id].invalidated_by = neighborhood_max_id;
                    protohits[protohit_id].valid = false;
                    *invalidated_protohits += 1;
                }
                visited[candidate_linear_index] = neighborhood_max_id;
                *changes += 1;
                
                // Expand the neighborhood, adopting the highest SNR protohit within the neighborhood if needed
                for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                    for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                        auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};

                        neighbor_coord.drift_index += drift_neighbor_offset;
                        neighbor_coord.frequency_channel += freq_neighbor_offset;

                        if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                                neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                            auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;
                            if (visited[linear_neighbor_index] >= 2) {
                                visited[linear_neighbor_index] = neighborhood_max_id;
                            }
                        }
                    }
                }

            }
        }
    }
}


__global__ void spread_components3_kernel(
    uint32_t *visited,
    int32_t number_drifts, int32_t number_channels,
    freq_drift_coord* neighbor_offsets,
    int neighborhood_size,
    device_protohit* protohits,
    int* number_protohits,
    int* invalidated_protohits,
    int* changes) {

    extern __shared__ char smem[];
    freq_drift_coord* s_neighbor_offsets = reinterpret_cast<freq_drift_coord*>(smem);
    for (int neighbor_coord_ind = threadIdx.x; neighbor_coord_ind < neighborhood_size; neighbor_coord_ind+=blockDim.x) {
        s_neighbor_offsets[neighbor_coord_ind] = neighbor_offsets[neighbor_coord_ind];
    }
    __syncthreads();

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;

    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto protohit_id = visited[candidate_linear_index];
        if (protohit_id >= 8) {
            if (protohits[protohit_id].valid) {
                // Look at every neighbor, check for any protohit with a higher SNR
                for (int neighbor_index=0; neighbor_index < neighborhood_size; neighbor_index += 1) {
                    auto neighbor_offset = s_neighbor_offsets[neighbor_index];
                    auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
                    neighbor_coord.drift_index += neighbor_offset.drift_index;
                    neighbor_coord.frequency_channel += neighbor_offset.frequency_channel;

                    if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                            neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {
                        auto neighbor_linear_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;
                        auto neighbor_id = visited[neighbor_linear_index];
                        if (neighbor_id >= 8 && neighbor_id != protohit_id) {
                            // These are both part of protohits, but not the same one

                            auto neighbor_protohit = protohits[neighbor_id];
                            if (neighbor_protohit.snr > protohits[protohit_id].snr) {
                                if (neighbor_protohit.valid) {
                                    protohits[protohit_id].invalidated_by = neighbor_id;
                                    protohits[protohit_id].valid = false;
                                    visited[neighbor_linear_index] = protohit_id;
                                    *invalidated_protohits += 1;
                                    *changes += 1;
                                } else {
                                    // This is an awkward case because we know these are both bad, but the other is greater
                                    // so take whatever it has been invalidated by
                                    protohits[protohit_id].invalidated_by = protohits[neighbor_id].invalidated_by;
                                    protohits[protohit_id].valid = false;
                                    visited[candidate_linear_index] = protohits[neighbor_id].invalidated_by;
                                    *invalidated_protohits += 1;
                                    *changes += 1;
                                }
                            } else {
                                // Our SNR is higher than the neighbor
                                if (neighbor_protohit.valid) {
                                    // The neighbor is lower, so we can invalidate it and set it
                                    protohits[neighbor_id].invalidated_by = protohit_id;
                                    protohits[neighbor_id].valid = false;
                                    visited[neighbor_linear_index] = protohit_id;
                                    *invalidated_protohits += 1;
                                    *changes += 1;
                                } else {
                                    // We're greater than the neighbor snr, but it's also marked as invalid
                                    // visited[neighbor_linear_index] = protohit_id;
                                    // *changes += 1;
                                }
                            }
                        } else if (neighbor_id >= 2 && neighbor_id < 8) {
                            // Doesn't belong to any valid protohit yet, but it's valid to spread
                            visited[neighbor_linear_index] = protohit_id;
                            *changes += 1;
                        }
                    }
                }
            } else {
                // belongs to an invalidated protohit, just update the index
                visited[candidate_linear_index] = protohits[protohit_id].invalidated_by;
                *changes += 1; // we *could* spread this... but let's not for now
            }
        } else if (protohit_id >= 2) {
            // This can be spread, but isn't above threshold. If a highest-SNR neighbor has a protohit id, we can adopt it
            double highest_snr_neighbor = -9999; // TODO....
            int best_protohit_id = 0;
            for (int neighbor_index=0; neighbor_index < neighborhood_size; neighbor_index += 1) {
                auto neighbor_offset = s_neighbor_offsets[neighbor_index];
                auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
                neighbor_coord.drift_index += neighbor_offset.drift_index;
                neighbor_coord.frequency_channel += neighbor_offset.frequency_channel;
                if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                    neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                    auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;

                    auto neighbor_id = visited[linear_neighbor_index];
                    if (neighbor_id >= 8 && protohits[neighbor_id].snr > highest_snr_neighbor) {
                        best_protohit_id = neighbor_id;
                        highest_snr_neighbor = protohits[neighbor_id].snr;
                    }
                }
            }
            if (best_protohit_id >= 8) {
                visited[candidate_linear_index] = best_protohit_id;
            }
        }
    }
}

__global__ void collect_protohit_md_kernel(
    uint8_t* rfi_low_sk,
    uint8_t* rfi_high_sk,
    uint32_t *visited,
    int32_t number_drifts, int32_t number_channels,
    device_protohit* protohits,
    int* number_protohits) {

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;

    for (int protohit_index = work_id; protohit_index < *number_protohits; protohit_index += increment_size) {
        auto this_protohit = protohits[protohit_index];
        if (this_protohit.valid) {
        auto index_max = this_protohit.index_max;

        auto drift_index = index_max.drift_index;
        auto frequency_channel = index_max.frequency_channel;


        int lower_freq_edge_index = frequency_channel;
        bool still_above_snr_2 = true;
        do {
            lower_freq_edge_index -= 1;
            auto linear_index = drift_index * number_channels + lower_freq_edge_index;
            still_above_snr_2 = visited[linear_index] > 1;
        } while(still_above_snr_2 && lower_freq_edge_index > 0);

        int upper_freq_edge_index = frequency_channel;
        still_above_snr_2 = true;
        do {
            upper_freq_edge_index += 1;
            auto linear_index = drift_index * number_channels + upper_freq_edge_index;
            still_above_snr_2 = visited[linear_index] > 1;
        } while(still_above_snr_2 && upper_freq_edge_index < number_channels-1);

        int linear_index = drift_index * number_channels + frequency_channel;
        this_protohit.low_sk_count = rfi_low_sk[linear_index];
        this_protohit.high_sk_count = rfi_high_sk[linear_index];
        this_protohit.binwidth = upper_freq_edge_index - lower_freq_edge_index;
        this_protohit.index_center = {.drift_index=drift_index, .frequency_channel=(lower_freq_edge_index + upper_freq_edge_index)/2};

        protohits[protohit_index] = this_protohit;
        // protohits[protohit_index].snr = candidate_snr;
        // protohits[protohit_index].max_integration = candidate_val;
        // protohits[protohit_index].desmeared_noise = noise_at_candidate_drift;
        }
    }
}

std::vector<protohit>
bliss::find_components_above_threshold_cuda(bland::ndarray                   doppler_spectrum,
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
    *dev_num_maxima = 8;

    uint32_t* visited;
    cudaMalloc((void**)&visited, sizeof(uint32_t) * number_drifts * number_channels);
    cudaMemset(visited, 0, number_drifts * number_channels);

    int number_blocks = 112;
    int block_size = 512;
    find_components_kernel<<<number_blocks, block_size>>>(
        doppler_spectrum_data,
        visited,
        thrust::raw_pointer_cast(dev_noise_per_drift.data()),
        number_drifts, number_channels,
        noise_floor, snr_threshold,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        dev_num_maxima
    );
    cudaDeviceSynchronize();

    int* u_number_changes;
    cudaMallocManaged(&u_number_changes, sizeof(int));
    *u_number_changes = 0;

    int* u_invalidated_protohits;
    cudaMallocManaged(&u_invalidated_protohits, sizeof(int));
    *u_invalidated_protohits = 0;

    block_size = 512;
    number_blocks = 112;
    do {
        *u_number_changes = 0;
        *u_invalidated_protohits = 0;

        spread_components_kernel<<<number_blocks, block_size>>>(
            visited,
            number_drifts, number_channels,
            neighbor_l1_dist,
            thrust::raw_pointer_cast(dev_protohits.data()),
            dev_num_maxima,
            u_invalidated_protohits,
            u_number_changes
        );
        cudaDeviceSynchronize();

    } while (*u_number_changes > 0);


    dev_protohits.erase(dev_protohits.begin(), dev_protohits.begin() + 8);
    *dev_num_maxima -= 8;
    dev_protohits.resize(*dev_num_maxima);

    collect_protohit_md_kernel<<<1, 1>>>(
        dedrifted_rfi.low_spectral_kurtosis.data_ptr<uint8_t>(),
        dedrifted_rfi.high_spectral_kurtosis.data_ptr<uint8_t>(),
        visited,
        number_drifts, number_channels,
        thrust::raw_pointer_cast(dev_protohits.data()),
        dev_num_maxima
    );
    cudaDeviceSynchronize();

    thrust::host_vector<device_protohit> host_protohits(dev_protohits.begin(), dev_protohits.end());

    cudaFree(dev_num_maxima);
    cudaFree(visited);
    cudaFree(u_number_changes);
    cudaFree(u_invalidated_protohits);

    std::vector<protohit> export_protohits;
    export_protohits.reserve(host_protohits.size());
    for (auto &simple_hit : host_protohits) {
        if (simple_hit.valid) {
            auto new_protohit = protohit();
            new_protohit.index_max = simple_hit.index_max;
            new_protohit.index_center = simple_hit.index_center;
            new_protohit.snr = simple_hit.snr;
            new_protohit.max_integration = simple_hit.max_integration;
            new_protohit.desmeared_noise = simple_hit.desmeared_noise;
            new_protohit.binwidth = simple_hit.binwidth;
            new_protohit.rfi_counts = {{
                {flag_values::low_spectral_kurtosis, simple_hit.low_sk_count},
                {flag_values::high_spectral_kurtosis, simple_hit.high_sk_count},
                {flag_values::filter_rolloff, 0},
            }};

            export_protohits.push_back(new_protohit);
        }
    }

    return export_protohits;
}
