
#include "local_maxima_cpu.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <stdexcept>

using namespace bliss;

std::vector<protohit>
bliss::find_local_maxima_above_threshold_cpu(bland::ndarray                  doppler_spectrum,
                                            integrated_flags                 dedrifted_rfi,
                                            float                            noise_floor,
                                            std::vector<protohit_drift_info> noise_per_drift,
                                            float                            snr_threshold,
                                            int                              neighbor_l1_dist) {

    auto doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto doppler_spectrum_strides = doppler_spectrum.strides();
    auto doppler_spectrum_shape = doppler_spectrum.shape();

    // Use 1 to mark visited, then we can potentially replace this creation with a mask of above thresh to speed things
    // up a bit
    auto visited =
            bland::ndarray(doppler_spectrum.shape(), 1, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

    auto visited_data    = visited.data_ptr<uint8_t>();
    auto visited_shape   = visited.shape();
    auto visited_strides = visited.strides();

    auto curr_coord = freq_drift_coord{.drift_index=0, .frequency_channel=0};

    auto numel = doppler_spectrum.numel();
    // how much greater must the local max be above the neighborhood (2%)
    // TODO: might be worth thinking through a distance that drops with L1/L2 increase
    constexpr float rtol = 1.0f;

    std::vector<protohit> maxima;
    for (int64_t n = 0; n < numel; ++n) {
        // 1. Check if this is not visited & above threshold
        auto linear_visited_index = curr_coord.drift_index * visited_strides[0] + curr_coord.frequency_channel * visited_strides[1];
        auto linear_doppler_spectrum_index = curr_coord.drift_index * doppler_spectrum_strides[0] + curr_coord.frequency_channel * doppler_spectrum_strides[1];
        if (visited_data[linear_visited_index] > 0) {
            // 2. Mark as visited
            visited_data[linear_visited_index] = 0; // We've been here now!

            // 3. Check that we're above our search threshold
            auto candidate_maxima_val = doppler_spectrum_data[linear_doppler_spectrum_index];
            auto noise_at_this_drift = noise_per_drift[curr_coord.drift_index].integration_adjusted_noise;
            auto candidate_snr = (candidate_maxima_val - noise_floor) / noise_at_this_drift;
            if (candidate_snr > snr_threshold) {
                // 4. Check if it is greater than surrounding neighborhood
                bool neighborhood_max = true;
                for (int freq_neighbor_offset=-neighbor_l1_dist; freq_neighbor_offset < neighbor_l1_dist; ++freq_neighbor_offset) {
                    for (int drift_neighbor_offset=-neighbor_l1_dist + abs(freq_neighbor_offset); drift_neighbor_offset < neighbor_l1_dist-abs(freq_neighbor_offset); ++drift_neighbor_offset) {
                        auto neighbor_coord = curr_coord;

                        neighbor_coord.drift_index += drift_neighbor_offset;
                        neighbor_coord.frequency_channel += freq_neighbor_offset;

                        // check if the next coordinate is valid and not visited
                        if (neighbor_coord.drift_index >= 0 && neighbor_coord.drift_index < doppler_spectrum_shape[0] &&
                            neighbor_coord.frequency_channel >= 0 && neighbor_coord.frequency_channel < doppler_spectrum_shape[1]) {
                        
                            auto linear_neighbor_index = neighbor_coord.drift_index * doppler_spectrum_strides[0] + neighbor_coord.frequency_channel * doppler_spectrum_strides[1];
                            auto neighbor_val = doppler_spectrum_data[linear_neighbor_index];
                            auto neighbor_noise = noise_per_drift[neighbor_coord.drift_index].integration_adjusted_noise;
                            auto neighbor_snr = (neighbor_val - noise_floor) / neighbor_noise;
                            if (candidate_snr > rtol * neighbor_snr) {
                                // we know this neighbor can't be a candidate maxima...
                                auto visited_linear_index = neighbor_coord.drift_index * visited_strides[0] + neighbor_coord.frequency_channel * visited_strides[1];
                                // visited_strider.to_linear_offset(neighbor_coord)
                                visited_data[visited_linear_index] = 0;
                            } else {
                                neighborhood_max = false;
                                break;
                            }
                        } else {
                            // the neighbor is outside of grid... this generally causes problems so abandon
                            neighborhood_max = false;
                            break;
                        }
                    }
                }

                // 3. Add to list of local maxima
                if (neighborhood_max) {
                    protohit c;
                    auto coor = freq_drift_coord{.drift_index=curr_coord.drift_index, .frequency_channel=curr_coord.frequency_channel};
                    c.index_max = coor;
                    c.snr = candidate_snr;
                    c.locations.push_back(coor);
                    c.max_integration = candidate_maxima_val;
                    c.desmeared_noise = noise_at_this_drift;

                    c.rfi_counts[flag_values::low_spectral_kurtosis] =
                            dedrifted_rfi.low_spectral_kurtosis.scalarize<uint8_t>({curr_coord.drift_index, curr_coord.frequency_channel});
                    c.rfi_counts[flag_values::high_spectral_kurtosis] =
                            dedrifted_rfi.high_spectral_kurtosis.scalarize<uint8_t>({curr_coord.drift_index, curr_coord.frequency_channel});
                    c.rfi_counts[flag_values::sigma_clip] =
                            dedrifted_rfi.sigma_clip.scalarize<uint8_t>({curr_coord.drift_index, curr_coord.frequency_channel});
                    // c.rfi_counts[flag_values::magnitude] =
                    //         dedrifted_rfi.magnitude.scalarize<uint8_t>(curr_coord);
                    // c.rfi_counts[flag_values::sigma_clip] =
                    //         dedrifted_rfi.sigma_clip.scalarize<uint8_t>(curr_coord);

                    // At the local max drift rate, look up and down in frequency channel adding locations that are
                    // above the SNR threshold AND continuing to decrease
                    auto max_lower_edge   = coor;
                    auto lower_edge_value = candidate_maxima_val;
                    bool expand_band_down = true;
                    do {
                        // WARN: this is the first time in this file we make an assumption about drift index position
                        max_lower_edge.frequency_channel -= 1;

                        // check if the next coordinate is valid
                        if (max_lower_edge.frequency_channel >= 0) {
                            auto linear_visited_index = max_lower_edge.drift_index * visited_strides[0] + max_lower_edge.frequency_channel * visited_strides[1];
                            auto linear_doppler_spectrum_index = max_lower_edge.drift_index * doppler_spectrum_strides[0] + max_lower_edge.frequency_channel * doppler_spectrum_strides[1];

                            auto new_lower_edge_value = doppler_spectrum_data[linear_doppler_spectrum_index];
                            // Keep expanding "bandwidth" at the drift of local max as long as extended bandwidth
                            // continues to decrease in magnitude and it's still a "hit" above SNR threshold
                            if ((new_lower_edge_value-noise_floor)/noise_at_this_drift > snr_threshold && new_lower_edge_value < lower_edge_value) {
                                visited_data[linear_visited_index] = 0; // We've been here now!
                                // It's tempting to mark this as "visited" so it won't be added to a different local
                                // maxima hit or used for its consideration, but to keep things reproducible (it doesn't
                                // matter which local maxima was looked at first), we'll not mark it as visited
                                lower_edge_value = new_lower_edge_value;
                                // c.locations.push_back(max_lower_edge);
                            } else {
                                expand_band_down = false;
                            }
                        } else {
                            expand_band_down = false; // we reached the edge of our spectra
                        }

                    } while (expand_band_down);

                    auto max_upper_edge   = coor; // TODO: check if this should be `coor` (reference extending lower edge)
                    auto upper_edge_value = candidate_maxima_val;
                    bool expand_band_up   = true;
                    do {
                        max_upper_edge.frequency_channel += 1;

                        // check if the next coordinate is valid and not visited
                        if (max_upper_edge.frequency_channel < visited_shape[1]) {
                            auto linear_visited_index = max_upper_edge.drift_index * visited_strides[0] + max_upper_edge.frequency_channel * visited_strides[1];
                            auto linear_doppler_spectrum_index = max_upper_edge.drift_index * doppler_spectrum_strides[0] + max_upper_edge.frequency_channel * doppler_spectrum_strides[1];

                            auto new_upper_edge_value = doppler_spectrum_data[linear_doppler_spectrum_index];
                            // Keep expanding "bandwidth" at the drift of local max as long as extended bandwidth
                            // continues to decrease in magnitude and it's still a "hit" above SNR threshold
                            // if (new_upper_edge_value > hard_threshold && new_upper_edge_value < upper_edge_value) {
                            if ((new_upper_edge_value-noise_floor)/noise_at_this_drift > snr_threshold && new_upper_edge_value < upper_edge_value) {
                                visited_data[linear_visited_index] = 0; // We've been here now!

                                upper_edge_value = new_upper_edge_value;
                                // c.locations.push_back(max_upper_edge);
                            } else {
                                expand_band_up = false;
                            }
                        } else {
                            expand_band_up = false; // we reached the edge of our spectra
                        }

                    } while (expand_band_up);

                    auto binwidth = max_upper_edge.frequency_channel - max_lower_edge.frequency_channel;
                    c.binwidth = binwidth;
                    c.index_center = {.drift_index=curr_coord.drift_index, .frequency_channel=(max_upper_edge.frequency_channel + max_lower_edge.frequency_channel)/2};
                    maxima.push_back(c);
                }
            }
        }

        // 4. Increment ndindex
        if (++curr_coord.frequency_channel == doppler_spectrum_shape[1]) {
            ++curr_coord.drift_index;
            curr_coord.frequency_channel = 0;
        }
    }
    return maxima;
}
