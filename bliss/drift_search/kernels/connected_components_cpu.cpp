#include "connected_components_cpu.hpp"

#include <drift_search/hit_search.hpp> // component

#include <bland/stride_helper.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <queue>

using namespace bliss;

std::vector<protohit> bliss::find_components_in_binary_mask_cpu(const bland::ndarray         &mask,
                                                                 std::vector<bland::nd_coords> neighborhood) {
    // We're going to change values, so get a copy
    auto threshold_mask = bland::copy(mask);
    // Thresholded_mask holds binary information on which bins passed a threshold. Group adjacent
    // bins that passed the threshold together (connected components)
    std::vector<protohit> components;

    auto                 thresholded_data    = threshold_mask.data_ptr<uint8_t>();
    auto                 thresholded_shape   = threshold_mask.shape();
    auto                 thresholded_strides = threshold_mask.strides();

    auto curr_coord = freq_drift_coord{.drift_index=0, .frequency_channel=0};

    auto numel = threshold_mask.numel();

    std::queue<freq_drift_coord> coord_queue;

    for (int64_t n = 0; n < numel; ++n) {
        // dereference threshold_mask w/ linear offset computed from current nd_index
        // auto curr_linear = strider.to_linear_offset(curr_coord);
        auto curr_linear = curr_coord.drift_index * thresholded_strides[0] + curr_coord.frequency_channel * thresholded_strides[1];
        if (thresholded_data[curr_linear] > 0) {
            coord_queue.push(curr_coord);
            protohit this_component;

            while (!coord_queue.empty()) {
                auto idx = coord_queue.front();
                coord_queue.pop();

                auto linear_index = idx.drift_index * thresholded_strides[0] + idx.frequency_channel * thresholded_strides[1];

                // Assume in bounds and if above threshold, add it to the protohit
                if (thresholded_data[linear_index] > 0) {
                    this_component.locations.push_back(idx);
                    thresholded_data[linear_index] = 0;

                    // Then add all of the neighbors as candidates
                    for (auto &neighbor_offset : neighborhood) {
                        auto neighbor_coord = idx;
                        neighbor_coord.drift_index += neighbor_offset[0];
                        neighbor_coord.frequency_channel += neighbor_offset[1];

                        // If this is in bounds, above threshold, not visited add it
                        if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < thresholded_shape[0] &&
                                neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < thresholded_shape[1]) {
                            // TODO: add back in that we have not visited this
                            coord_queue.push(neighbor_coord);
                        }
                    }
                } else {
                    // Otherwise, toss it away
                    continue;
                }
            }

            components.push_back(this_component); // Assuming 's' is some statistic you compute for each protohit
        }

        if (++curr_coord.frequency_channel == thresholded_shape[1]) {
            ++curr_coord.drift_index;
            curr_coord.frequency_channel = 0;
        }
    }
    return components;
}

std::vector<protohit>
bliss::find_components_above_threshold_cpu(bland::ndarray                    doppler_spectrum,
                                            integrated_flags                 dedrifted_rfi,
                                            float                            noise_floor,
                                            std::vector<protohit_drift_info> noise_per_drift,
                                            float                            snr_threshold,
                                            int                              neighbor_l1_dist) {

    std::vector<protohit> components;

    if (doppler_spectrum.dtype() != bland::ndarray::datatype::float32) {
        throw std::runtime_error("find_components_above_threshold: dedrifted doppler spectrum was not float. Only cpu "
                                 "float is supported right now");
    }
    auto doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto doppler_spectrum_strides = doppler_spectrum.strides();
    auto doppler_spectrum_shape = doppler_spectrum.shape();

    auto visited =
            bland::ndarray(doppler_spectrum.shape(), 0, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

    auto visited_data    = visited.data_ptr<uint8_t>();
    auto visited_shape   = visited.shape();
    auto visited_strides = visited.strides();

    auto curr_coord = freq_drift_coord{.drift_index=0, .frequency_channel=0};

    std::queue<freq_drift_coord> coord_queue;

    auto numel = doppler_spectrum.numel();
    for (int64_t n = 0; n < numel; ++n) {
        // Compute linear offsets for current location to search
        auto visited_linear = curr_coord.drift_index * visited_strides[0] + curr_coord.frequency_channel * visited_strides[1];
        auto doppler_spectrum_linear = curr_coord.drift_index * doppler_spectrum_strides[0] + curr_coord.frequency_channel * doppler_spectrum_strides[1];

        // If not visited and signal is above threshold...
        auto this_drift_noise = noise_per_drift[curr_coord.drift_index].integration_adjusted_noise;
        auto candidate_snr = (doppler_spectrum_data[doppler_spectrum_linear] - noise_floor) / this_drift_noise;
        if (visited_data[visited_linear] == 0 && candidate_snr > snr_threshold) {
            coord_queue.push(curr_coord);
            protohit this_component;
            this_component.max_integration = 0;
            this_component.snr = std::numeric_limits<float>::min();

            while (!coord_queue.empty()) {
                auto idx = coord_queue.front();
                coord_queue.pop();

                auto this_coord_visited_linear = idx.drift_index * visited_strides[0] + idx.frequency_channel * visited_strides[1];
                auto this_coord_doppler_spectrum_linear = idx.drift_index * doppler_spectrum_strides[0] + idx.frequency_channel * doppler_spectrum_strides[1];

                // Test if this drift is part of the current cluster:
                // * we have not visited this yet
                // * it passes the hard threshold
                // TODO, add some more to greedily merge clusters split by noise / minor signal power drops at off
                // integrations by testing a distance metric We might even want to pass a callable if we can define this
                // well (and expose to python as callable!)
                // Assume in bounds and if above threshold, add it to the protohit
                if (visited_data[this_coord_visited_linear] == 0) {
                    ++visited_data[this_coord_visited_linear]; // Mark as visited

                    auto this_coord_val = doppler_spectrum_data[this_coord_doppler_spectrum_linear];
                    auto this_drift_noise = noise_per_drift[idx.drift_index].integration_adjusted_noise;
                    auto this_coord_snr = (this_coord_val - noise_floor) / this_drift_noise;

                    if (this_coord_snr > snr_threshold/2) {
                        // Track some stats for this cluster like the maximum value and where it is
                        if (this_coord_snr > this_component.snr) {
                            this_component.index_max = idx;
                            this_component.max_integration = this_coord_val;
                            this_component.desmeared_noise = this_drift_noise;
                            this_component.snr = this_coord_snr;

                            // auto dedrifted_rfi = drift_plane.integrated_rfi();

                            this_component.rfi_counts[flag_values::low_spectral_kurtosis] =
                                    dedrifted_rfi.low_spectral_kurtosis.scalarize<uint8_t>({curr_coord.drift_index, curr_coord.frequency_channel});
                            this_component.rfi_counts[flag_values::high_spectral_kurtosis] =
                                    dedrifted_rfi.high_spectral_kurtosis.scalarize<uint8_t>({curr_coord.drift_index, curr_coord.frequency_channel});
                            this_component.rfi_counts[flag_values::filter_rolloff] =
                                    dedrifted_rfi.filter_rolloff.scalarize<uint8_t>({curr_coord.drift_index, curr_coord.frequency_channel});
                            // this_component.rfi_counts[flag_values::magnitude] =
                            //         dedrifted_rfi.magnitude.scalarize<uint8_t>(curr_coord);
                            // this_component.rfi_counts[flag_values::sigma_clip] =
                            //         dedrifted_rfi.sigma_clip.scalarize<uint8_t>(curr_coord);
                        }

                        this_component.locations.push_back(idx);

                        // Then add all of the neighbors as candidates
                        for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                            for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                                auto neighbor_coord = idx;

                                neighbor_coord.drift_index += drift_neighbor_offset;
                                neighbor_coord.frequency_channel += freq_neighbor_offset;

                                // If this is in bounds, not visited add it
                                if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < visited_shape[0]
                                        && neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < visited_shape[1]) {
                                    auto neighbor_visited_linear = neighbor_coord.drift_index * visited_strides[0] + neighbor_coord.frequency_channel * visited_strides[1];
                                    if (visited_data[neighbor_visited_linear] == 0) {
                                        coord_queue.push(neighbor_coord);
                                    }
                                }
                            }
                        }
                    } else {
                        // Otherwise, toss it away
                        continue;
                    }
                }
            }

            // We have the index_max and max. Use that to come up with binwidth and index_center
            auto max_drift_index = this_component.index_max.drift_index;
            int64_t lower_freq_index_at_rate = this_component.index_max.frequency_channel;
            int64_t upper_freq_index_at_rate = this_component.index_max.frequency_channel;
            for (const auto &l : this_component.locations) {
                if (l.drift_index == max_drift_index) {
                    if (l.frequency_channel > upper_freq_index_at_rate) {
                        upper_freq_index_at_rate = l.frequency_channel;
                    } else if (l.frequency_channel < lower_freq_index_at_rate) {
                        lower_freq_index_at_rate = l.frequency_channel;
                    }
                }
            }
            this_component.binwidth = (upper_freq_index_at_rate - lower_freq_index_at_rate);
            this_component.index_center = {.drift_index=max_drift_index, .frequency_channel=(upper_freq_index_at_rate + lower_freq_index_at_rate)/2};

            components.push_back(this_component);
        }

        // Increment the nd_index
        if (++curr_coord.frequency_channel == visited_shape[1]) {
            ++curr_coord.drift_index;
            curr_coord.frequency_channel = 0;
        }
    }
    return components;
}
