
#include <drift_search/local_maxima.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <stdexcept>

using namespace bliss;

// Helper to abstract out increments
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

std::vector<component> bliss::find_local_maxima_above_threshold(coarse_channel        &dedrifted_coarse_channel,
                                                                float                  snr_threshold,
                                                                std::vector<nd_coords> max_neighborhood) {
    auto noise_stats = dedrifted_coarse_channel.noise_estimate();
    
    auto drift_plane = dedrifted_coarse_channel.integrated_drift_plane();
    auto integration_length = drift_plane.integration_steps();

    const auto noise_and_thresholds_per_drift = compute_noise_and_snr_thresholds(noise_stats, integration_length, drift_plane.drift_rate_info(), snr_threshold);

    std::vector<component> maxima;
    
    auto doppler_spectrum = drift_plane.integrated_drift_plane();
    if (doppler_spectrum.dtype() != bland::ndarray::datatype::float32) {
        throw std::runtime_error(
                "find_local_maxima_above_threshold: dedrifted doppler spectrum was not float. Only cpu "
                "float is supported right now");
    }
    auto          doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto          doppler_spectrum_strides = doppler_spectrum.strides();
    stride_helper doppler_spectrum_strider(doppler_spectrum.shape(), doppler_spectrum_strides);

    fmt::print("doppler_spectrum is on device {}\n", doppler_spectrum.device().repr());


    // Use 1 to mark visited, then we can potentially replace this creation with a mask of above thresh to speed things
    // up a bit
    auto visited = bland::ndarray(doppler_spectrum.shape(),
                                  1,
                                  bland::ndarray::datatype::uint8,
                                  bland::ndarray::dev::cpu);
    bland::fill(visited, 1);
    auto          visited_data    = visited.data_ptr<uint8_t>();
    auto          visited_shape   = visited.shape();
    auto          visited_strides = visited.strides();
    stride_helper visited_strider(visited_shape, visited_strides);
    nd_coords     curr_coord(visited_shape.size(), 0);

    auto numel = visited.numel();
    // how much greater must the local max be above the neighborhood (2%)
    // TODO: might be worth thinking through a distance that drops with L1/L2 increase
    constexpr float rtol = 1.0f;

    for (int64_t n = 0; n < numel; ++n) {
        // 1. Check if this is not visited & above threshold
        auto linear_visited_index          = visited_strider.to_linear_offset(curr_coord);
        auto linear_doppler_spectrum_index = doppler_spectrum_strider.to_linear_offset(curr_coord);
        if (visited_data[linear_visited_index] > 0) {
            // 2. Mark as visited
            visited_data[linear_visited_index] = 0; // We've been here now!

            // 3. Check that we're above our search threshold
            auto candidate_maxima_val = doppler_spectrum_data[linear_doppler_spectrum_index];
            auto hard_threshold = noise_and_thresholds_per_drift[curr_coord[0]].first;
            if (candidate_maxima_val > hard_threshold) {

                // 4. Check if it is greater than surrounding neighborhood
                bool neighborhood_max = true;
                for (auto &neighbor_offset : max_neighborhood) {
                    bool in_bounds      = true;
                    auto neighbor_coord = curr_coord;
                    for (int dim = 0; dim < visited_shape.size(); ++dim) {
                        neighbor_coord[dim] += neighbor_offset[dim];
                        if (neighbor_coord[dim] < 0 || neighbor_coord[dim] >= visited_shape[dim]) {
                            in_bounds = false;
                            break;
                        }
                    }

                    // check if the next coordinate is valid and not visited
                    if (in_bounds) {
                        auto linear_neighbor_index = doppler_spectrum_strider.to_linear_offset(neighbor_coord);
                        if (candidate_maxima_val > rtol * doppler_spectrum_data[linear_neighbor_index]) {
                            // we know this neighbor can't be a candidate maxima...
                            visited_data[visited_strider.to_linear_offset(neighbor_coord)] = 0;
                        } else {
                            neighborhood_max = false;
                            break;
                        }
                    }
                }

                // 3. Add to list of local maxima
                if (neighborhood_max) {
                    component c;
                    c.index_max = curr_coord;
                    c.locations.push_back(curr_coord);
                    c.max_integration = candidate_maxima_val;
                    c.desmeared_noise = noise_and_thresholds_per_drift[curr_coord[0]].second;

                    auto dedrifted_rfi = drift_plane.integrated_rfi();

                    c.rfi_counts[flag_values::low_spectral_kurtosis] =
                            dedrifted_rfi.low_spectral_kurtosis.scalarize<uint8_t>(curr_coord);
                    c.rfi_counts[flag_values::high_spectral_kurtosis] =
                            dedrifted_rfi.high_spectral_kurtosis.scalarize<uint8_t>(curr_coord);
                    c.rfi_counts[flag_values::filter_rolloff] =
                            dedrifted_rfi.filter_rolloff.scalarize<uint8_t>(curr_coord);
                    // c.rfi_counts[flag_values::magnitude] =
                    //         dedrifted_rfi.magnitude.scalarize<uint8_t>(curr_coord);
                    // c.rfi_counts[flag_values::sigma_clip] =
                    //         dedrifted_rfi.sigma_clip.scalarize<uint8_t>(curr_coord);

                    // Wow, this very conceptually simple thing of adding a bandwidth estimate to local maxima hits
                    // wound up being some ugly code... rethink it

                    // At the local max drift rate, look up and down in frequency channel adding locations that are
                    // above the SNR threshold AND continuing to decrease
                    auto max_lower_edge   = curr_coord;
                    auto lower_edge_value = candidate_maxima_val;
                    bool expand_band_down = true;
                    do {
                        // WARN: this is the first time in this file we make an assumption about drift index position
                        max_lower_edge[1] -= 1;

                        bool lower_extension_in_bounds = true;
                        // This chunk is copypasta from above, consider refactoring to reduce
                        for (int dim = 0; dim < visited_shape.size(); ++dim) {
                            if (max_lower_edge[dim] < 0 || max_lower_edge[dim] >= visited_shape[dim]) {
                                lower_extension_in_bounds = false;
                                break;
                            }
                        }

                        // check if the next coordinate is valid
                        if (lower_extension_in_bounds) {
                            auto linear_visited_index = visited_strider.to_linear_offset(max_lower_edge);
                            auto linear_doppler_spectrum_index =
                                    doppler_spectrum_strider.to_linear_offset(max_lower_edge);
                            auto new_lower_edge_value = doppler_spectrum_data[linear_doppler_spectrum_index];
                            // Keep expanding "bandwidth" at the drift of local max as long as extended bandwidth
                            // continues to decrease in magnitude and it's still a "hit" above SNR threshold
                            if (new_lower_edge_value > snr_threshold && new_lower_edge_value < lower_edge_value) {
                                // It's tempting to mark this as "visited" so it won't be added to a different local
                                // maxima hit or used for its consideration, but to keep things reproducible (it doesn't
                                // matter which local maxima was looked at first), we'll not mark it as visited
                                lower_edge_value = new_lower_edge_value;
                                c.locations.push_back(max_lower_edge);
                            } else {
                                expand_band_down = false;
                            }
                        } else {
                            expand_band_down = false; // we reached the edge of our spectra
                        }

                    } while (expand_band_down);

                    bool expand_band_up   = true;
                    auto max_upper_edge   = curr_coord;
                    auto upper_edge_value = candidate_maxima_val;
                    do {
                        // WARN: this is the first time in this file we make an assumption about drift index position
                        max_upper_edge[1] += 1;

                        bool upper_extension_in_bounds = true;
                        // This chunk is copypasta from above, consider refactoring to reduce
                        for (int dim = 0; dim < visited_shape.size(); ++dim) {
                            if (max_upper_edge[dim] < 0 || max_upper_edge[dim] >= visited_shape[dim]) {
                                upper_extension_in_bounds = false;
                                break;
                            }
                        }

                        // check if the next coordinate is valid and not visited
                        if (upper_extension_in_bounds) {
                            auto linear_visited_index = visited_strider.to_linear_offset(max_upper_edge);
                            auto linear_doppler_spectrum_index =
                                    doppler_spectrum_strider.to_linear_offset(max_upper_edge);
                            auto new_upper_edge_value = doppler_spectrum_data[linear_doppler_spectrum_index];
                            // Keep expanding "bandwidth" at the drift of local max as long as extended bandwidth
                            // continues to decrease in magnitude and it's still a "hit" above SNR threshold
                            if (new_upper_edge_value > snr_threshold && new_upper_edge_value < upper_edge_value) {
                                upper_edge_value = new_upper_edge_value;
                                c.locations.push_back(max_upper_edge);
                            } else {
                                expand_band_up = false;
                            }
                        } else {
                            expand_band_up = false; // we reached the edge of our spectra
                        }

                    } while (expand_band_up);

                    // TODO: this doesn't have the concept of a "component" the same way connected_components does... do
                    // we care?
                    maxima.push_back(c);
                }
            }
        }

        // 4. Increment ndindex
        // TODO: I decided this might be useful to add to the strider class...
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
    return maxima;
}
