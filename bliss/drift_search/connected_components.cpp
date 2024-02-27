
#include <drift_search/connected_components.hpp>
#include <drift_search/hit_search.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <queue>

using namespace bliss;

// Helper to abstract out increments (this is copied in local_maxima, TODO: factor out to bland utils)
struct stride_helper {

    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    stride_helper(std::vector<int64_t> shape, std::vector<int64_t> stride) : shape(shape), stride(stride) {}

    int64_t to_linear_offset(const nd_coords &coords) const {
        // TODO compute linear index of curr_coords from shape and stride
        auto linear_offset = 0;
        for (int dim = 0; dim < shape.size(); ++dim) {
            linear_offset += coords[dim] * stride[dim]; // TODO: also add offset
        }
        return linear_offset;
    }
};

std::vector<component> bliss::find_components_in_binary_mask(const bland::ndarray  &mask,
                                                             std::vector<nd_coords> neighborhood) {
    // We're going to change values, so get a copy
    auto threshold_mask = bland::copy(mask);
    if (neighborhood.empty()) {
        neighborhood = {
                {-1, 0},
                {1, 0},
                {0, -1},
                {0, 1},
        };
    }
    // Thresholded_mask holds binary information on which bins passed a threshold. Group adjacent
    // bins that passed the threshold together (connected components)
    std::vector<component> components;

    auto          thresholded_data    = threshold_mask.data_ptr<uint8_t>();
    auto          thresholded_shape   = threshold_mask.shape();
    auto          thresholded_strides = threshold_mask.strides();
    stride_helper strider(thresholded_shape, thresholded_strides);
    nd_coords     curr_coord(thresholded_shape.size(), 0);

    auto numel = threshold_mask.numel();

    std::queue<nd_coords> coord_queue;

    for (int64_t n = 0; n < numel; ++n) {
        // dereference threshold_mask w/ linear offset computed from current nd_index
        auto curr_linear = strider.to_linear_offset(curr_coord);
        if (thresholded_data[curr_linear] > 0) {
            coord_queue.push(curr_coord);
            component this_component;

            while (!coord_queue.empty()) {
                nd_coords idx = coord_queue.front();
                coord_queue.pop();

                auto linear_index = strider.to_linear_offset(idx);

                // Assume in bounds and if above threshold, add it to the component
                if (thresholded_data[linear_index] > 0) {
                    this_component.locations.push_back(idx);
                    thresholded_data[linear_index] = 0;

                    // Then add all of the neighbors as candidates
                    for (auto &neighbor_offset : neighborhood) {
                        bool in_bounds      = true;
                        auto neighbor_coord = idx;
                        for (int dim = 0; dim < thresholded_shape.size(); ++dim) {
                            neighbor_coord[dim] += neighbor_offset[dim];
                            if (neighbor_coord[dim] < 0 || neighbor_coord[dim] >= thresholded_shape[dim]) {
                                in_bounds = false;
                                break;
                            }
                        }

                        // If this is in bounds, above threshold, not visited add it
                        if (in_bounds) {
                            coord_queue.push(neighbor_coord);
                        }
                    }
                } else {
                    // Otherwise, toss it away
                    continue;
                }
            }

            components.push_back(this_component); // Assuming 's' is some statistic you compute for each component
        }
        // Increment the nd_index
        for (int dim = curr_coord.size() - 1; dim >= 0; --dim) {
            // If we're not at the end of this dim, keep going
            if (++curr_coord[dim] != thresholded_shape[dim]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                curr_coord[dim] = 0;
            }
        }
    }
    return components;
}

std::vector<component> bliss::find_components_above_threshold(coarse_channel        &dedrifted_coarse_channel,
                                                              float                  snr_threshold,
                                                              std::vector<nd_coords> neighborhood) {
    auto noise_stats = dedrifted_coarse_channel.noise_estimate();

    if (neighborhood.empty()) {
        neighborhood = {
                {-1, 0},
                {1, 0},
                {0, -1},
                {0, 1},
        };
    }

    auto drift_plane = dedrifted_coarse_channel.integrated_drift_plane();
    auto integration_length = drift_plane._integration_steps;
    auto noise_and_thresholds_per_drift = compute_noise_and_snr_thresholds(noise_stats, integration_length, drift_plane._drift_rate_info, snr_threshold);

    std::vector<component> components;

    auto doppler_spectrum = drift_plane._integrated_drifts;
    if (doppler_spectrum.dtype() != bland::ndarray::datatype::float32) {
        throw std::runtime_error("find_components_above_threshold: dedrifted doppler spectrum was not float. Only cpu "
                                 "float is supported right now");
    }
    auto          doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto          doppler_spectrum_strides = doppler_spectrum.strides();
    stride_helper doppler_spectrum_strider(doppler_spectrum.shape(), doppler_spectrum_strides);

    auto          visited         = bland::ndarray(doppler_spectrum.shape(),
                                  0,
                                  bland::ndarray::datatype::uint8,
                                  bland::ndarray::dev::cpu);
    auto          visited_data    = visited.data_ptr<uint8_t>();
    auto          visited_shape   = visited.shape();
    auto          visited_strides = visited.strides();
    stride_helper visited_strider(visited_shape, visited_strides);
    nd_coords     curr_coord(visited_shape.size(), 0);

    auto numel = visited.numel();

    std::queue<nd_coords> coord_queue;

    for (int64_t n = 0; n < numel; ++n) {
        // Compute linear offsets for current location to search
        auto visited_linear          = visited_strider.to_linear_offset(curr_coord);
        auto doppler_spectrum_linear = doppler_spectrum_strider.to_linear_offset(curr_coord);
        // If not visited and signal is above threshold...
        auto hard_threshold = noise_and_thresholds_per_drift[curr_coord[0]].first;
        if (visited_data[visited_linear] == 0 && doppler_spectrum_data[doppler_spectrum_linear] > hard_threshold) {
            coord_queue.push(curr_coord);
            component this_component;
            this_component.max_integration = 0;

            while (!coord_queue.empty()) {
                nd_coords idx = coord_queue.front();
                coord_queue.pop();

                auto this_coord_visited_linear          = visited_strider.to_linear_offset(idx);
                auto this_coord_doppler_spectrum_linear = doppler_spectrum_strider.to_linear_offset(idx);
                // Test if this drift is part of the current cluster:
                // * we have not visited this yet
                // * it passes the hard threshold
                // TODO, add some more to greedily merge clusters split by noise / minor signal power drops at off
                // integrations by testing a distance metric We might even want to pass a callable if we can define this
                // well (and expose to python as callable!)
                // Assume in bounds and if above threshold, add it to the component
                if (visited_data[this_coord_visited_linear] == 0 &&
                    doppler_spectrum_data[this_coord_doppler_spectrum_linear] > hard_threshold) {

                    this_component.locations.push_back(idx);
                    ++visited_data[this_coord_visited_linear]; // Mark as visited

                    // Track some stats for this cluster like the maximum value and where it is
                    if (doppler_spectrum_data[this_coord_doppler_spectrum_linear] > this_component.max_integration) {
                        this_component.max_integration = doppler_spectrum_data[this_coord_doppler_spectrum_linear];
                        this_component.desmeared_noise = noise_and_thresholds_per_drift[curr_coord[0]].second;

                        this_component.index_max       = idx;
                        this_component.rfi_counts[flag_values::low_spectral_kurtosis] =
                                drift_plane._dedrifted_rfi.low_spectral_kurtosis.scalarize<uint8_t>(curr_coord);
                        this_component.rfi_counts[flag_values::high_spectral_kurtosis] =
                                drift_plane._dedrifted_rfi.high_spectral_kurtosis.scalarize<uint8_t>(curr_coord);
                        this_component.rfi_counts[flag_values::filter_rolloff] =
                                drift_plane._dedrifted_rfi.filter_rolloff.scalarize<uint8_t>(curr_coord);
                        this_component.rfi_counts[flag_values::magnitude] =
                                drift_plane._dedrifted_rfi.magnitude.scalarize<uint8_t>(curr_coord);
                        this_component.rfi_counts[flag_values::sigma_clip] =
                                drift_plane._dedrifted_rfi.sigma_clip.scalarize<uint8_t>(curr_coord);
                    }
                    // Then add all of the neighbors as candidates
                    for (auto &neighbor_offset : neighborhood) {
                        bool in_bounds      = true;
                        auto neighbor_coord = idx;
                        for (int dim = 0; dim < visited_shape.size(); ++dim) {
                            neighbor_coord[dim] += neighbor_offset[dim];
                            if (neighbor_coord[dim] < 0 || neighbor_coord[dim] >= visited_shape[dim]) {
                                in_bounds = false;
                                break;
                            }
                        }

                        // If this is in bounds, above threshold, not visited add it
                        if (in_bounds) {
                            coord_queue.push(neighbor_coord);
                        }
                    }
                } else {
                    // Otherwise, toss it away
                    continue;
                }
            }

            components.push_back(this_component); // Assuming 's' is some statistic you compute for each component
        }
        // Increment the nd_index
        for (int dim = curr_coord.size() - 1; dim >= 0; --dim) {
            // If we're not at the end of this dim, keep going
            if (++curr_coord[dim] != visited_shape[dim]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                curr_coord[dim] = 0;
            }
        }
    }
    return components;
}
