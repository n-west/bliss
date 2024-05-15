#include "connected_components_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


__global__ void initialize_components_kernel(float* doppler_spectrum_data,
    uint32_t *g_visited,
    protohit_drift_info* noise_per_drift,
    int32_t number_drifts, int32_t number_channels,
    float noise_floor,
    float snr_threshold,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits,
    int* m_non_core_labels) {

    // Strategy:
    // 1) grid-strided loop to label each bin in freq-drift plane as a local_maxima with a unique id, above SNR threshold,
    // above SNR/2 threshold, or not above SNR threshold
    // 2) while changed > 0:
    //   A) grid-stride loop and replace the neighborhood with the highest index of the neighborhood

    // g_visited possible values:
    // 0: not g_visited
    // 1: g_visited, not above SNR threshold
    // 2: g_visited, above SNR/2
    // 3: g_visited, above SNR
    // >=8: g_visited, above SNR and local_maxima

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;
    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        if (g_visited[candidate_linear_index] == 0) {
            g_visited[candidate_linear_index] = 1;
            auto candidate_val = doppler_spectrum_data[candidate_linear_index];
            auto noise_at_candidate_drift = noise_per_drift[candidate_drift_index].integration_adjusted_noise;
            auto candidate_snr = (candidate_val - noise_floor) / noise_at_candidate_drift;
            if (candidate_snr > snr_threshold) {
                // g_visited[candidate_linear_index] = 3;

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
                    // TODO: sanity check that number_protohits is actually below the max number of protohits we allocated space for
                    auto protohit_index = atomicAdd(number_protohits, 1);
                    protohits[protohit_index].index_max = {.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
                    protohits[protohit_index].snr = candidate_snr;
                    protohits[protohit_index].max_integration = candidate_val;
                    protohits[protohit_index].desmeared_noise = noise_at_candidate_drift;
                    protohits[protohit_index].invalidated_by = 0;

                    g_visited[candidate_linear_index] = protohit_index;
                } else {
                    // These need a label but aren't true hits and don't have allocated memory for them!
                    auto nonprotohit_label = atomicAdd(m_non_core_labels, 1);
                    g_visited[candidate_linear_index] = nonprotohit_label;
                }

            }
            // TODO: come back to this, we might just want to make this a non-core label
            //  else if (candidate_snr > snr_threshold/2) {
            //     g_visited[candidate_linear_index] = 2;
            // }
        } else {

        }
    }
}

__device__ int bfs_greedy_search_connected_protohit(uint32_t *g_visited,
    int number_drifts, int number_channels,
    freq_drift_coord central_point,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits,
    int first_noncore_label) {

    bool found_real_protohit = false;
    float best_real_protohit_snr = -9999;
    int best_protohit_id = 0;
    // Check all of the neighbors for the highest SNR protohit
    for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
        for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
            auto neighbor_coord = central_point;
            neighbor_coord.drift_index += drift_neighbor_offset;
            neighbor_coord.frequency_channel += freq_neighbor_offset;

            if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                    neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;

                auto neighbor_id = g_visited[linear_neighbor_index];
                if (neighbor_id >= first_noncore_label && !found_real_protohit) {
                    // we haven't found a core point yet so we can merge to the lower label
                    best_protohit_id = min(best_protohit_id, neighbor_id);
                } else if (neighbor_id >= 8 && neighbor_id < first_noncore_label) {
                    auto neighbor_snr = protohits[neighbor_id].snr;
                    if (!found_real_protohit || neighbor_snr > best_real_protohit_snr) {
                        best_real_protohit_snr = neighbor_snr;
                        best_protohit_id = neighbor_id;
                        found_real_protohit = true;
                    }
                }
            }
        }
    }
    // We now want to expand our perimeter somehow. At our neighborhood edges look at their neighbors but we also don't need to bother
    // looking back at places we've already looked
    if (found_real_protohit) {
        return best_protohit_id;
    } else {
        return -1;
    }
}

__global__ void scan_connected_component(uint32_t *g_visited,
    int32_t number_drifts, int32_t number_channels,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits,
    int first_noncore_label) {

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;

    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto protohit_id = g_visited[candidate_linear_index];

        if (protohit_id >= 8) {
            // Scan the neighborhood for a better label
            auto neighborhood_anchor = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
            auto connected_protohit = bfs_greedy_search_connected_protohit(g_visited, number_drifts, number_channels, neighborhood_anchor, neighbor_l1_dist, protohits, number_protohits, first_noncore_label);
            if (connected_protohit >= 8) {
                g_visited[candidate_linear_index] = connected_protohit;
            }
        }
    }
}

__global__ void analyze_labels_kernel(
    uint32_t *g_visited,
    int32_t number_drifts, int32_t number_channels,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits) {

    // Grid-stride loop to invalidate our neighbor in favor of a component in our neighborhood with higher SNR
    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;

    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto protohit_id = g_visited[candidate_linear_index];

        if (protohit_id >= 8) {
            int neighborhood_best_component = protohit_id;
            float neighborhood_best_snr = protohits[protohit_id].snr;

            bool oob = false;
            auto central_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
            // Look in our neighborhood for the protohit_id with the highest snr, and track its id
            for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist && !oob; ++freq_neighbor_offset) {
                auto neighbor_coord_freq = central_coord;
                neighbor_coord_freq.frequency_channel += freq_neighbor_offset;
                if (neighbor_coord_freq.frequency_channel < 0 || neighbor_coord_freq.frequency_channel >= number_channels) {
                    // a neighbor is out of bounds, so axe it
                    neighborhood_best_component = protohit_id;
                    oob = true;
                    break;
                }
                for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset) && !oob; ++drift_neighbor_offset) {
                    auto neighbor_coord_drift = neighbor_coord_freq;
                    neighbor_coord_drift.drift_index += drift_neighbor_offset;

                    if (neighbor_coord_drift.drift_index < 0 || neighbor_coord_drift.drift_index >= number_drifts) {
                        // a neighbor is out of bounds, so axe it
                        neighborhood_best_component = protohit_id;
                        oob = true;
                        break;
                    }

                    auto linear_neighbor_index = neighbor_coord_drift.drift_index * number_channels + neighbor_coord_drift.frequency_channel;
                    auto neighbor_id = g_visited[linear_neighbor_index];
                    if (neighbor_id >= 8) {
                        auto neighbor_snr = protohits[neighbor_id].snr;
                        if (neighbor_snr > neighborhood_best_snr) {
                            neighborhood_best_snr = neighbor_snr;
                            neighborhood_best_component = neighbor_id;
                        }
                        // if (neighborhood_best_snr < 1000) {
                        //     neighborhood_best_component = 1;
                        // } else {
                        //     printf("centered at (%lld, %lld): at neighbor %lld, %lld the SNR was %f\n", central_coord.drift_index, central_coord.frequency_channel, neighbor_coord_drift.drift_index, neighbor_coord_drift.frequency_channel, neighbor_snr);
                        // }
                    }
                }
            }

            if (!oob && neighborhood_best_component != protohit_id) {
                protohits[protohit_id].invalidated_by = neighborhood_best_component;
            //     atomicCAS(&(protohits[protohit_id].invalidated_by),
            //         0,
            //         neighborhood_best_component);
            }
        }
    }
}

__global__ void collect_protohit_md_kernel(
    uint8_t* rfi_low_sk,
    uint8_t* rfi_high_sk,
    uint32_t *g_visited,
    int32_t number_drifts, int32_t number_channels,
    device_protohit* protohits,
    int* number_protohits) {

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;

    for (int protohit_index = work_id; protohit_index < *number_protohits; protohit_index += increment_size) {
        auto this_protohit = protohits[protohit_index];
        if (this_protohit.invalidated_by == 0) {
            auto index_max = this_protohit.index_max;

            auto drift_index = index_max.drift_index;
            auto frequency_channel = index_max.frequency_channel;


            int lower_freq_edge_index = frequency_channel;
            bool still_above_snr_2 = true;
            do {
                lower_freq_edge_index -= 1;
                auto linear_index = drift_index * number_channels + lower_freq_edge_index;
                still_above_snr_2 = g_visited[linear_index] > 1;
            } while(still_above_snr_2 && lower_freq_edge_index > 0);

            int upper_freq_edge_index = frequency_channel;
            still_above_snr_2 = true;
            do {
                upper_freq_edge_index += 1;
                auto linear_index = drift_index * number_channels + upper_freq_edge_index;
                still_above_snr_2 = g_visited[linear_index] > 1;
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
    int number_protohits = numel / (neighbor_l1_dist*neighbor_l1_dist);
    thrust::device_vector<device_protohit> dev_protohits(number_protohits);

    // All malloc's in one place to make it easier to track down free's later on
    int* m_num_maxima;
    cudaMallocManaged(&m_num_maxima, sizeof(int));
    *m_num_maxima = 8;

    int first_noncore_label = number_protohits;
    int* m_non_core_labels;
    cudaMallocManaged(&m_non_core_labels, sizeof(int));
    *m_non_core_labels = first_noncore_label;

    uint32_t* g_visited;
    cudaMalloc((void**)&g_visited, sizeof(uint32_t) * number_drifts * number_channels);
    cudaMemset(g_visited, 0, sizeof(uint32_t) * number_drifts * number_channels);

    // Label equivalence algorithm:
    // 1) Initialize: Initialize labels for all pixels
    // 2) Scan: Construct label-equivalence list based on connectivity (what is the best label that each label is equivalent to)
    // 3) Analysis: Resolve equivalence (replace with the end of the equivalence list)


    // Step 1: Initialize labels
    // Each pixel looks within its neighborhood to determine if it is the neighborhood maxima AND above SNR threshold. If so, it gets a unique
    // component (pre-protohit) label id and fills in appropriate metadata
    // If a pixel is not a local maxima  but is above the threshold the label id 2 is given
    // Proposal: if a pixel is above the threshold but not a local maxima, assign a psuedo-label that can be correctly
    // merged. It would be easiest if this is actually allocated a device_protohit to fill in a correct "invalidated_by" field
    //
    // We have ~60 B / dev_protohit and we should be able to have a maximum of channels * time / sizeof(neighborhood) local maxima but the
    // practical limit is likely much lower than that. In order to make space for these other dumb protohits
    int number_blocks = 112;
    int block_size = 512;
    initialize_components_kernel<<<number_blocks, block_size>>>(
        doppler_spectrum_data,
        g_visited,
        thrust::raw_pointer_cast(dev_noise_per_drift.data()),
        number_drifts, number_channels,
        noise_floor, snr_threshold,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima,
        m_non_core_labels
    );
    auto launch_ret = cudaDeviceSynchronize();
    auto kernel_ret = cudaGetLastError();
    if (launch_ret != cudaSuccess) {
        fmt::print("initialize_components_kernel: cuda launch got error {} ({})\n", launch_ret, cudaGetErrorString(launch_ret));
    }
    if (kernel_ret != cudaSuccess) {
        fmt::print("initialize_components_kernel: cuda kernel got error {} ({})\n", kernel_ret, cudaGetErrorString(kernel_ret));
    }

    // Step 2: Scan
    // Each pixel with a protohit id looks at its neighborhood to see if a better protohit (higher SNR) is neighboring it, which invalidates
    // this protohit by the neighboring protohit with better SNR. Use AtomicCAS to swap invalidated_by if our SNR is higher than the current
    // invalidated_by SNR-- that might also require us to invalidate the old node by us as well
    // At the end of this, there should be a fixed number of "valid" protohits that is deterministic but with non-deterministic ids
    block_size = 1;
    number_blocks = 1;
    scan_connected_component<<<number_blocks, block_size>>>(
        g_visited,
        number_drifts, number_channels,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima,
        first_noncore_label
    );

    launch_ret = cudaDeviceSynchronize();
    kernel_ret = cudaGetLastError();
    if (launch_ret != cudaSuccess) {
        fmt::print("scan_connected_component: cuda launch got error {} ({})\n", launch_ret, cudaGetErrorString(launch_ret));
    }
    if (kernel_ret != cudaSuccess) {
        fmt::print("scan_connected_component: cuda kernel got error {} ({})\n", kernel_ret, cudaGetErrorString(kernel_ret));
    }

    // Step 3: Analysis
    // For each pixel, follow the chain of "invalidated by" kernels to update all labels
    analyze_labels_kernel<<<112, 512>>>(
        g_visited,
        number_drifts, number_channels,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima
    );

    launch_ret = cudaDeviceSynchronize();
    kernel_ret = cudaGetLastError();
    if (launch_ret != cudaSuccess) {
        fmt::print("analyze_labels_kernel: cuda launch got error {} ({})\n", launch_ret, cudaGetErrorString(launch_ret));
    }
    if (kernel_ret != cudaSuccess) {
        fmt::print("analyze_labels_kernel: cuda kernel got error {} ({})\n", kernel_ret, cudaGetErrorString(kernel_ret));
    }

    dev_protohits.erase(dev_protohits.begin(), dev_protohits.begin() + 8);
    *m_num_maxima -= 8;
    fmt::print("dev_protohits.size = {} and m_num_maxima= {}\n", dev_protohits.size(), *m_num_maxima);
    dev_protohits.resize(*m_num_maxima); // this is probably a do-nothing

    collect_protohit_md_kernel<<<1, 1>>>(
        dedrifted_rfi.low_spectral_kurtosis.data_ptr<uint8_t>(),
        dedrifted_rfi.high_spectral_kurtosis.data_ptr<uint8_t>(),
        g_visited,
        number_drifts, number_channels,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima
    );
    cudaDeviceSynchronize();

    thrust::host_vector<device_protohit> host_protohits(dev_protohits.begin(), dev_protohits.end());

    cudaFree(m_num_maxima);
    cudaFree(m_non_core_labels);
    cudaFree(g_visited);

    std::vector<protohit> export_protohits;
    export_protohits.reserve(host_protohits.size());
    for (auto &simple_hit : host_protohits) {
        if (simple_hit.invalidated_by == 0) {
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
