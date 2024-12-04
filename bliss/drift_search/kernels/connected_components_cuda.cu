#include "connected_components_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;


__global__ void initialize_components_kernel(float* doppler_spectrum_data,
    uint32_t *g_labels,
    protohit_drift_info* noise_per_drift,
    int32_t number_drifts, int32_t number_channels,
    float noise_floor,
    float snr_threshold,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits,
    int n_max_protohits) {

    // Strategy:
    // 1) grid-strided loop to visit each pixel. For each pixel:
    //   1a) if below threshold: leave 0
    //   1b) if above threshold: look at neighborhood SNRs
    //   If this is the neighborhood max, then grab a new unique id and fill in protohit details
    //   If this is not the neighborhood max, set to a fake label id that is based on the linear index of neighborhood max


    // g_visited possible values after this routine:
    // 0: below threshold
    // >=8 && < n_max_protohits: local maxima
    // > n_max_protohits: non-maximum but points to index of the max of this neighborhood

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;
    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto candidate_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto candidate_val = doppler_spectrum_data[candidate_linear_index];
        auto noise_at_candidate_drift = noise_per_drift[candidate_drift_index].integration_adjusted_noise;
        auto candidate_snr = (candidate_val - noise_floor) / noise_at_candidate_drift;
        auto neighborhood_max_snr = candidate_snr;
        auto neighborhood_max_index = candidate_linear_index;
        if (candidate_snr > snr_threshold/2) {
            // Look at the entire neighborhood
            bool is_neighborhood_max = true;
            for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                    auto neighbor_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};

                    neighbor_coord.drift_index += drift_neighbor_offset;
                    neighbor_coord.frequency_channel += freq_neighbor_offset;

                    // Validate the bounds are within tolerance
                    if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < number_drifts &&
                            neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < number_channels) {

                        auto linear_neighbor_index = neighbor_coord.drift_index * number_channels + neighbor_coord.frequency_channel;
                        auto neighbor_val = doppler_spectrum_data[linear_neighbor_index];
                        auto neighbor_noise = noise_per_drift[neighbor_coord.drift_index].integration_adjusted_noise;
                        auto neighbor_snr = (neighbor_val - noise_floor) / neighbor_noise;
                        if (neighbor_snr > neighborhood_max_snr) {
                            is_neighborhood_max = false;
                            neighborhood_max_snr = neighbor_snr;
                            neighborhood_max_index = linear_neighbor_index;
                        }
                    } else {
                        // The neighborhood is out of bounds, so we won't let this be a neighborhood max
                        is_neighborhood_max = false;
                        goto done; // Multi-break required
                    }
                }
            }
            if (is_neighborhood_max) {
                if (candidate_snr > snr_threshold) {
                    // Get a unique label id and fill in details
                    // TODO: sanity check that number_protohits is actually below the max number of protohits we allocated space for
                    auto protohit_index = atomicAdd(number_protohits, 1);
                    protohits[protohit_index].index_max = {.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
                    protohits[protohit_index].snr = candidate_snr;
                    protohits[protohit_index].max_integration = candidate_val;
                    protohits[protohit_index].desmeared_noise = noise_at_candidate_drift;
                    protohits[protohit_index].invalidated_by = 0;

                    g_labels[candidate_linear_index] = protohit_index;
                } else {
                    // Nothing needs to happen here, but this could happen because
                    // this pixel is the center and above snr_threshold/2
                }
            } else {
                // Assign to a unique label id that doesn't get metadata, but is based on neighborhood max value
                g_labels[candidate_linear_index] = n_max_protohits + neighborhood_max_index;
            }
            done: ; // empty statement to avoid a warning
        }

    }
}

__device__ int find_root(uint32_t *g_labels,
    int linear_index,
    int first_nonroot_label) {

    auto component = g_labels[linear_index];
    // Keep climbing to subsequently higher maxima until reaching a local maxima which is the root of this graph
    while (component > first_nonroot_label) {
        auto node_address = component - first_nonroot_label;
        component = g_labels[node_address];
    }
    return component;
}


__global__ void resolve_labels(uint32_t *g_labels,
    int32_t number_drifts, int32_t number_channels,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits,
    int first_nonroot_label) {

    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;

    for (int n = work_id; n < numel; n += increment_size) {

        // For each pixel, hill climb to find the root node (which is a legitimate local_maxima)
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto protohit_id = g_labels[linear_index];

        if (protohit_id >= 8) {
            if (protohit_id > first_nonroot_label) {
                auto root = find_root(g_labels, linear_index, first_nonroot_label);
                if (root < first_nonroot_label) {
                    g_labels[linear_index] = root;
                } else {
                    // If our snr was above snr_threshold/2 but below snr and we couldn't connect
                    // to a valid protohit, womp womp-- not part of a hit
                    g_labels[linear_index] = 0;
                }
            }
        }
    }
}

__global__ void merge_labels(
    uint32_t *g_labels,
    int32_t number_drifts, int32_t number_channels,
    int neighbor_l1_dist,
    device_protohit* protohits,
    int* number_protohits) {
    // Every node points to a valid protohit, now make every node that is not a component max point
    // towards component max

    // Grid-stride loop to invalidate our neighbor in favor of a component in our neighborhood with higher SNR
    auto work_id = blockDim.x * blockIdx.x + threadIdx.x;
    auto increment_size = blockDim.x * gridDim.x;
    auto numel = number_drifts * number_channels;

    for (int n = work_id; n < numel; n += increment_size) {
        auto candidate_frequency_channel = n % number_channels;
        auto candidate_drift_index = n / number_channels;

        auto center_linear_index = candidate_drift_index * number_channels + candidate_frequency_channel;
        auto protohit_id = g_labels[center_linear_index];

        if (protohit_id >= 8) {
            auto pixel_root = protohit_id;
            // Chase to the current root
            while (protohits[pixel_root].invalidated_by != 0) {
                pixel_root = protohits[pixel_root].invalidated_by;
            }

            auto central_coord = freq_drift_coord{.drift_index=candidate_drift_index, .frequency_channel=candidate_frequency_channel};
            // Look in our neighborhood for the protohit_id with the highest snr, and track its id
            for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                auto neighbor_coord_freq = central_coord;
                neighbor_coord_freq.frequency_channel += freq_neighbor_offset;
                if (neighbor_coord_freq.frequency_channel < 0 || neighbor_coord_freq.frequency_channel >= number_channels) {
                    // a neighbor is out of bounds, don't pay attention to this (shouldn't even get here)
                    break;
                }
                for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                    auto neighbor_coord_drift = neighbor_coord_freq;
                    neighbor_coord_drift.drift_index += drift_neighbor_offset;

                    if (neighbor_coord_drift.drift_index < 0 || neighbor_coord_drift.drift_index >= number_drifts) {
                        // a neighbor is out of bounds, don't pay attention to this (shouldn't even get here)
                        break;
                    }

                    auto neighbor_linear_index = neighbor_coord_drift.drift_index * number_channels + neighbor_coord_drift.frequency_channel;
                    auto neighbor_id = g_labels[neighbor_linear_index];

                    if (neighbor_id >= 8) {
                        // Go to the neighbor's root
                        auto neighbor_root = neighbor_id;

                        int old = 0;
                        int merged_root = pixel_root;

                        do {
                            // Find the root of our pixel and the neighbor
                            while (protohits[pixel_root].invalidated_by != 0) {
                                pixel_root = protohits[pixel_root].invalidated_by;
                            }
                            while (protohits[neighbor_root].invalidated_by != 0) {
                                neighbor_root = protohits[neighbor_root].invalidated_by;
                            }
                            if (pixel_root == neighbor_root) {
                                break;
                            }

                            // Between the two roots, find the one with better SNR and merge the other to this graph
                            int to_invalidate = pixel_root;
                            if (protohits[pixel_root].snr > protohits[neighbor_root].snr) {
                                to_invalidate = neighbor_root;
                                merged_root = pixel_root;
                            } else if (protohits[neighbor_root].snr > protohits[pixel_root].snr) {
                                to_invalidate = pixel_root;
                                merged_root = neighbor_root;
                            } else {
                                // the equality condition very rarely can hit. This is known to be possible only at coarse channel
                                // boundaries with a DC tone right on the edge that crosses above SNR threshold. In this case,
                                // they should both eventually be invalidated by a closer to DC tone that captures more of the
                                // signal energy, just merge with a consistent rule: prefer lower linear index
                                auto neighbor_root_index = protohits[neighbor_root].index_max;
                                auto pixel_root_index = protohits[pixel_root].index_max;
                                if (neighbor_root_index.drift_index * number_channels + neighbor_root_index.frequency_channel < pixel_root_index.drift_index * number_channels + pixel_root_index.frequency_channel) {
                                    to_invalidate = pixel_root;
                                    merged_root = neighbor_root;
                                } else {
                                    to_invalidate = neighbor_root;
                                    merged_root = pixel_root;
                                }
                            }

                            // Try the (atomic) swap. The current invalidated_field must be 0
                            old = atomicCAS(
                                &(protohits[to_invalidate].invalidated_by),
                                0,
                                merged_root
                            );

                            // TODO: path compression

                        } while (old != 0);

                    }
                }
            }

        }
    }
}

__global__ void collect_protohit_md_kernel(
    uint8_t* rfi_low_sk,
    uint8_t* rfi_high_sk,
    uint32_t *g_labels,
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
                still_above_snr_2 = g_labels[linear_index] > 1;
            } while(still_above_snr_2 && lower_freq_edge_index > 0);

            int upper_freq_edge_index = frequency_channel;
            still_above_snr_2 = true;
            do {
                upper_freq_edge_index += 1;
                auto linear_index = drift_index * number_channels + upper_freq_edge_index;
                still_above_snr_2 = g_labels[linear_index] > 1;
            } while(still_above_snr_2 && upper_freq_edge_index < number_channels-1);

            int linear_index = drift_index * number_channels + frequency_channel;
            this_protohit.low_sk_count = rfi_low_sk[linear_index];
            this_protohit.high_sk_count = rfi_high_sk[linear_index];
            this_protohit.binwidth = upper_freq_edge_index - lower_freq_edge_index;
            this_protohit.index_center = {.drift_index=drift_index, .frequency_channel=(lower_freq_edge_index + upper_freq_edge_index)/2};

            protohits[protohit_index] = this_protohit;
        }
    }
}


template <typename T>
thrust::device_vector<T> safe_device_vector(int64_t size) {
    try {
        return thrust::device_vector<T>(size);
    }
    catch (thrust::system::detail::bad_alloc &e) {
        fmt::print("ERROR: while allocating safe vector: {}\n", e.what());
        throw std::runtime_error("GPU Memory allocation failed while running find_components");
    }
}
template <typename T, typename It>
thrust::device_vector<T> safe_device_vector(It begin_alloc, It end_alloc) {
    try {
        return thrust::device_vector<T>(begin_alloc, end_alloc);
    }
    catch (thrust::system::detail::bad_alloc &e) {
        fmt::print("ERROR: while allocating safe vector: {}\n", e.what());
        throw std::runtime_error("GPU Memory allocation failed while running find_components");
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

    auto dev_noise_per_drift = safe_device_vector<protohit_drift_info>(noise_per_drift.begin(), noise_per_drift.end());
    // We can only possibly have one max for every neighborhood, so that's a reasonably efficient max neighborhood
    int number_protohits = numel / (neighbor_l1_dist*neighbor_l1_dist);
    auto dev_protohits = safe_device_vector<device_protohit>(number_protohits);

    // All malloc's in one place to make it easier to track down free's later on
    int* m_num_maxima;
    auto malloc_ret = cudaMallocManaged(&m_num_maxima, sizeof(int));
    *m_num_maxima = 8;
    if (malloc_ret != cudaSuccess) {
        // Clean up our existing allocation
        fmt::print("ERROR: allocating space for m_num_maxima: cudaMalloc({}) got error {} ({})\n", sizeof(uint32_t) * number_drifts * number_channels, (int)malloc_ret, cudaGetErrorString(malloc_ret));
        throw std::runtime_error("find_components_above_threshold_cuda did not have enough vRAM to continue");
    }

    int first_noncore_label = number_protohits;

    uint32_t* g_labels;
    malloc_ret = cudaMalloc((void**)&g_labels, sizeof(uint32_t) * number_drifts * number_channels);
    if (malloc_ret != cudaSuccess) {
        // Clean up our existing allocation
        cudaFree(m_num_maxima);
        fmt::print("ERROR: allocating space for labels: cudaMalloc({}) got error {} ({})\n", sizeof(uint32_t) * number_drifts * number_channels, (int)malloc_ret, cudaGetErrorString(malloc_ret));
        throw std::runtime_error("find_components_above_threshold_cuda did not have enough vRAM to continue");
    }
    malloc_ret = cudaMemset(g_labels, 0, sizeof(uint32_t) * number_drifts * number_channels);
    if (malloc_ret != cudaSuccess) {
        fmt::print("initializing labels: cudaMemset got error {} ({})\n", (int)malloc_ret, cudaGetErrorString(malloc_ret));
    }

    // Step 1: Initialize labels
    // Each pixel looks within its neighborhood to determine if it is the neighborhood maxima AND above SNR threshold. If so, it gets a unique
    // component (pre-protohit) label id and fills in appropriate metadata
    // If a pixel is not a local maxima  but is above the threshold the label id 2 is given
    // Proposal: if a pixel is above the threshold but not a local maxima, assign a psuedo-label that can be correctly
    // merged. It would be easiest if this is actually allocated a device_protohit to fill in a correct "invalidated_by" field
    //
    // We have ~60 B / dev_protohit and we should be able to have a maximum of channels * time / sizeof(neighborhood) local maxima but the
    // practical limit is likely much lower than that. In order to make space for these other dumb protohits
    int number_blocks = 4096;
    int block_size = 64;
    initialize_components_kernel<<<number_blocks, block_size>>>(
        doppler_spectrum_data,
        g_labels,
        thrust::raw_pointer_cast(dev_noise_per_drift.data()),
        number_drifts, number_channels,
        noise_floor, snr_threshold,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima,
        first_noncore_label
    );
    // auto launch_ret = cudaDeviceSynchronize();
    // auto kernel_ret = cudaGetLastError();
    // if (launch_ret != cudaSuccess) {
    //     fmt::print("initialize_components_kernel: cuda launch got error {} ({})\n", (int)launch_ret, cudaGetErrorString(launch_ret));
    // }
    // if (kernel_ret != cudaSuccess) {
    //     fmt::print("initialize_components_kernel: cuda kernel got error {} ({})\n", (int)kernel_ret, cudaGetErrorString(kernel_ret));
    // }

    // Step 2: Resolve
    // Each pixel with a protohit id looks at its neighborhood to see if a better protohit (higher SNR) is neighboring it, which invalidates
    // this protohit by the neighboring protohit with better SNR. Use AtomicCAS to swap invalidated_by if our SNR is higher than the current
    // invalidated_by SNR-- that might also require us to invalidate the old node by us as well
    // At the end of this, there should be a fixed number of "valid" protohits that is deterministic but with non-deterministic ids
    resolve_labels<<<number_blocks, block_size>>>(
        g_labels,
        number_drifts, number_channels,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima,
        first_noncore_label
    );
    // launch_ret = cudaDeviceSynchronize();
    // kernel_ret = cudaGetLastError();
    // if (launch_ret != cudaSuccess) {
    //     fmt::print("resolve_labels: cuda launch got error {} ({})\n", (int)launch_ret, cudaGetErrorString(launch_ret));
    // }
    // if (kernel_ret != cudaSuccess) {
    //     fmt::print("resolve_labels: cuda kernel got error {} ({})\n", (int)kernel_ret, cudaGetErrorString(kernel_ret));
    // }

    // Step 3: Analysis
    // For each pixel, follow the chain of "invalidated by" kernels to update all labels
    merge_labels<<<number_blocks, block_size>>>(
        g_labels,
        number_drifts, number_channels,
        neighbor_l1_dist,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima
    );
    // launch_ret = cudaDeviceSynchronize();
    // kernel_ret = cudaGetLastError();
    // if (launch_ret != cudaSuccess) {
    //     fmt::print("merge_labels: cuda launch got error {} ({})\n", (int)launch_ret, cudaGetErrorString(launch_ret));
    // }
    // if (kernel_ret != cudaSuccess) {
    //     fmt::print("merge_labels: cuda kernel got error {} ({})\n", (int)kernel_ret, cudaGetErrorString(kernel_ret));
    // }

    dev_protohits.erase(dev_protohits.begin(), dev_protohits.begin() + 8);
    *m_num_maxima -= 8;
    dev_protohits.resize(*m_num_maxima);

    collect_protohit_md_kernel<<<1, 512>>>(
        dedrifted_rfi.low_spectral_kurtosis.data_ptr<uint8_t>(),
        dedrifted_rfi.high_spectral_kurtosis.data_ptr<uint8_t>(),
        g_labels,
        number_drifts, number_channels,
        thrust::raw_pointer_cast(dev_protohits.data()),
        m_num_maxima
    );
    cudaDeviceSynchronize();

    thrust::host_vector<device_protohit> host_protohits(dev_protohits.begin(), dev_protohits.end());

    cudaFree(m_num_maxima);
    cudaFree(g_labels);

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
