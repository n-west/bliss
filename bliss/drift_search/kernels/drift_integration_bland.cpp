
#include "drift_integration_bland.hpp"

#include "core/frequency_drift_plane.hpp"

#include <bland/bland.hpp>

#include <fmt/format.h>
#include <fmt/core.h>

#include <cmath> // std::round, std::abs


using namespace bliss;

/**
 * Naive approach following a line through spectrum grid using
 * round-away-from-zero (commercial rounding) based on a slope of time span over
 * frequency span where the time span is always the full time extent and the
 * frequency span is the distance between the start and end of the linear drift.
 *
 * Note that if there are 8 time rows, the time span is 7. Likewise, the 8
 * drifts will have frequency spans of 0, 1, 2, 3, 4, 5, 6, 7 giving 8 slopes of
 * value 0/7, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7.
 */

constexpr bool collect_rfi = true;
[[nodiscard]] frequency_drift_plane
bliss::integrate_linear_rounded_bins_bland(bland::ndarray spectrum_grid,
                                     bland::ndarray       rfi_mask,
                                     integrate_drifts_options    options) {

    auto number_drifts = (options.high_rate - options.low_rate) / options.rate_step_size;
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;

    int32_t time_steps      = spectrum_grid.size(0);
    auto number_channels = spectrum_grid.size(1);

    auto maximum_drift_span = time_steps - 1;

    bland::ndarray drift_plane({number_drifts, number_channels}, spectrum_grid.dtype(), spectrum_grid.device());
    auto           rfi_in_drift = integrated_flags(number_drifts, number_channels, rfi_mask.device());

    bland::fill(drift_plane, 0.0f);

    auto first_channel = 0; // This needs to be incrememnted by the offset from the most negative drift
    int  print_count   = 0;
    // We use all time available inside this function
    for (int drift_index = 0; drift_index < number_drifts; ++drift_index) {
        // Drift in number of channels over the entire time extent
        auto drift_channels = options.low_rate + drift_index * options.rate_step_size;
        frequency_drift_plane::drift_rate rate;
        rate.index_in_plane = drift_index;

        // The actual slope of that drift (number channels / time)
        auto m = static_cast<float>(drift_channels) / static_cast<float>(maximum_drift_span);
        rate.drift_rate_slope = m;
        // If a single time step crosses more than 1 channel, there is smearing over multiple channels
        auto smeared_channels = std::round(std::abs(m));

        int desmear_bandwidth = 1;
        if (options.desmear) {
            desmear_bandwidth = std::max(1.0f, smeared_channels);
            rate.desmeared_bins = smeared_channels;
        }
        drift_rate_info.push_back(rate);

        for (int32_t t = 0; t < time_steps; ++t) {
            int freq_offset_at_time = std::round(m * t);

            for (int desmear_channel = 0; desmear_channel < desmear_bandwidth; ++desmear_channel) {

                if (m >= 0) {
                    // The accumulator (drift spectrum) stays fixed at 0 while the spectrum start increments
                    auto channel_offset = freq_offset_at_time + desmear_channel;

                    int64_t drift_freq_slice_start = 0;
                    int64_t drift_freq_slice_end   = number_channels - channel_offset;

                    int64_t spectrum_freq_slice_start = channel_offset;
                    int64_t spectrum_freq_slice_end   = number_channels;

                    auto drift_slice = bland::slice(drift_plane,
                                                    bland::slice_spec{0, drift_index, drift_index + 1},
                                                    bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});

                    auto spectrum_slice =
                            bland::slice(spectrum_grid,
                                         bland::slice_spec{0, t, t + 1},
                                         bland::slice_spec{1, spectrum_freq_slice_start, spectrum_freq_slice_end});

                    if (collect_rfi) {
                        // Keep track of how much each type of RFI is present along the drift track
                        // The slicing is there because at the edges we just need to trim the out of bounds cases
                        auto rfi_mask_slice =
                                bland::slice(rfi_mask,
                                             bland::slice_spec{0, t, t + 1},
                                             bland::slice_spec{1, spectrum_freq_slice_start, spectrum_freq_slice_end});

                        // TODO: think through bitshifting and underflow to see if this can just be a bitshift
                        auto filter_rolloff_slice =
                                bland::slice(rfi_in_drift.filter_rolloff,
                                             bland::slice_spec{0, drift_index, drift_index + 1},
                                             bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        filter_rolloff_slice = filter_rolloff_slice +
                                               (rfi_mask_slice & static_cast<uint8_t>(flag_values::filter_rolloff)) /
                                                       static_cast<uint8_t>(flag_values::filter_rolloff);

                        auto low_spectral_kurtosis_slice =
                                bland::slice(rfi_in_drift.low_spectral_kurtosis,
                                             bland::slice_spec{0, drift_index, drift_index + 1},
                                             bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        low_spectral_kurtosis_slice =
                                low_spectral_kurtosis_slice +
                                (rfi_mask_slice & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) /
                                        static_cast<uint8_t>(flag_values::low_spectral_kurtosis);

                        auto high_spectral_kurtosis_slice =
                                bland::slice(rfi_in_drift.high_spectral_kurtosis,
                                             bland::slice_spec{0, drift_index, drift_index + 1},
                                             bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        high_spectral_kurtosis_slice =
                                high_spectral_kurtosis_slice +
                                (rfi_mask_slice & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) /
                                        static_cast<uint8_t>(flag_values::high_spectral_kurtosis);

                        // auto magnitude_slice =
                        //         bland::slice(rfi_in_drift.magnitude,
                        //                      bland::slice_spec{0, drift_index, drift_index + 1},
                        //                      bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        // magnitude_slice =
                        //         magnitude_slice + (rfi_mask_slice & static_cast<uint8_t>(flag_values::magnitude)) /
                        //                                   static_cast<uint8_t>(flag_values::magnitude);

                        // auto sigma_clip_slice =
                        //         bland::slice(rfi_in_drift.sigma_clip,
                        //                      bland::slice_spec{0, drift_index, drift_index + 1},
                        //                      bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        // sigma_clip_slice =
                        //         sigma_clip_slice + (rfi_mask_slice & static_cast<uint8_t>(flag_values::sigma_clip)) /
                        //                                    static_cast<uint8_t>(flag_values::sigma_clip);
                    }

                    drift_slice = drift_slice + spectrum_slice / desmear_bandwidth;
                } else {
                    // At a negative drift rate, everything needs to scooch up instead of chop down
                    // the desmeared channel might need to be negative as well (it's the channel we're advancing
                    // towards)
                    auto channel_offset = freq_offset_at_time - desmear_channel;

                    int64_t drift_freq_slice_start = -drift_channels;
                    int64_t drift_freq_slice_end   = number_channels;

                    int64_t spectrum_freq_slice_start = -drift_channels + channel_offset;
                    int64_t spectrum_freq_slice_end   = number_channels + channel_offset;
                    if (spectrum_freq_slice_start < 0) {
                        // This condition occurs at fast drive rates (very negative) with desmearing on.
                        auto offset_amount = spectrum_freq_slice_start;

                        spectrum_freq_slice_start -= offset_amount;
                        drift_freq_slice_start -= offset_amount;
                    }

                    auto drift_slice = bland::slice(drift_plane,
                                                    bland::slice_spec{0, drift_index, drift_index + 1},
                                                    bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                    auto spectrum_slice =
                            bland::slice(spectrum_grid,
                                         bland::slice_spec{0, t, t + 1},
                                         bland::slice_spec{1, spectrum_freq_slice_start, spectrum_freq_slice_end});

                    if (collect_rfi) {
                        // Keep track of how much each type of RFI is present along the drift track
                        // The slicing is there because at the edges we just need to trim the out of bounds cases
                        auto rfi_mask_slice =
                                bland::slice(rfi_mask,
                                             bland::slice_spec{0, t, t + 1},
                                             bland::slice_spec{1, spectrum_freq_slice_start, spectrum_freq_slice_end});

                        // TODO: think through bitshifting and underflow to see if this can just be a bitshift
                        auto filter_rolloff_slice =
                                bland::slice(rfi_in_drift.filter_rolloff,
                                             bland::slice_spec{0, drift_index, drift_index + 1},
                                             bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});

                        filter_rolloff_slice = filter_rolloff_slice +
                                               (rfi_mask_slice & static_cast<uint8_t>(flag_values::filter_rolloff)) /
                                                       static_cast<uint8_t>(flag_values::filter_rolloff);


                        auto low_spectral_kurtosis_slice =
                                bland::slice(rfi_in_drift.low_spectral_kurtosis,
                                             bland::slice_spec{0, drift_index, drift_index + 1},
                                             bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});

                        low_spectral_kurtosis_slice =
                                low_spectral_kurtosis_slice +
                                (rfi_mask_slice & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) /
                                        static_cast<uint8_t>(flag_values::low_spectral_kurtosis);

                        auto high_spectral_kurtosis_slice =
                                bland::slice(rfi_in_drift.high_spectral_kurtosis,
                                             bland::slice_spec{0, drift_index, drift_index + 1},
                                             bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});

                        high_spectral_kurtosis_slice =
                                high_spectral_kurtosis_slice +
                                (rfi_mask_slice & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) /
                                        static_cast<uint8_t>(flag_values::high_spectral_kurtosis);

                        // auto magnitude_slice =
                        //         bland::slice(rfi_in_drift.magnitude,
                        //                      bland::slice_spec{0, drift_index, drift_index + 1},
                        //                      bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        // magnitude_slice =
                        //         magnitude_slice + (rfi_mask_slice & static_cast<uint8_t>(flag_values::magnitude)) /
                        //                                   static_cast<uint8_t>(flag_values::magnitude);

                        // auto sigma_clip_slice =
                        //         bland::slice(rfi_in_drift.sigma_clip,
                        //                      bland::slice_spec{0, drift_index, drift_index + 1},
                        //                      bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                        // sigma_clip_slice =
                        //         sigma_clip_slice + (rfi_mask_slice & static_cast<uint8_t>(flag_values::sigma_clip)) /
                        //                                    static_cast<uint8_t>(flag_values::sigma_clip);
                    }

                    drift_slice = drift_slice + spectrum_slice / desmear_bandwidth;
                }
            }
        }
    }

    // normalize back by integration length
    frequency_drift_plane freq_drift(drift_plane / time_steps, rfi_in_drift, time_steps, drift_rate_info);
    return freq_drift;
}

bland::ndarray bliss::integrate_linear_rounded_bins_bland(bland::ndarray     spectrum_grid,
                                                    integrate_drifts_options options) {
    auto dummy_rfi_mask                      = bland::ndarray({1, 1});
    auto drift_plane = integrate_linear_rounded_bins_bland(spectrum_grid, dummy_rfi_mask, options);
    return drift_plane.integrated_drift_plane();
}
