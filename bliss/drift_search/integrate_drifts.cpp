
#include "drift_search/integrate_drifts.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/bland.hpp>

using namespace bliss;

namespace detail {
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
[[nodiscard]] bland::ndarray integrate_linear_rounded_bins(const bland::ndarray    &spectrum_grid,
                                                           integrate_drifts_options options) {
    auto number_drifts = (options.high_rate - options.low_rate) / options.rate_step_size;

    auto time_steps      = spectrum_grid.size(0);
    auto number_channels = spectrum_grid.size(1);

    auto maximum_drift_span = time_steps - 1;

    bland::ndarray drift_plane({number_drifts, number_channels}, spectrum_grid.dtype(), spectrum_grid.device());
    bland::fill(drift_plane, 0.0f);

    auto first_channel = 0; // This needs to be incrememnted by the offset from the most negative drift
    int print_count = 0;
    // We use all time available inside this function
    for (int drift_index = 0; drift_index < number_drifts; ++drift_index) {
        // Drift in number of channels over the entire time extent
        auto drift_channels = options.low_rate + drift_index * options.rate_step_size;

        // The actual slope of that drift (number channels / time)
        auto m = static_cast<float>(drift_channels) / static_cast<float>(maximum_drift_span);
        // If a single time step crosses more than 1 channel, there is smearing over multiple channels
        auto smeared_channels = std::round(std::abs(m));

        int desmear_bandwidth = 1;
        if (options.desmear) {
            desmear_bandwidth = std::max(1.0f, smeared_channels);
        }
        fmt::print("drift step {} (m={})has {} smeared channels, so {} number_integrated_channels\n",
                   drift_channels,
                   m,
                   smeared_channels,
                   desmear_bandwidth);

        // these don't take in to account smear
        for (int t = 0; t < time_steps; ++t) {
            int freq_offset_at_time = std::round(m * t);

            for (int desmear_channel = 0; desmear_channel < desmear_bandwidth; ++desmear_channel) {

                if (m >= 0) {
                    // The accumulator (drift spectrum) stays fixed at 0 while the spectrum start increments
                    auto channel_offset = freq_offset_at_time + desmear_channel;

                    int64_t drift_freq_slice_start    = 0;
                    int64_t drift_freq_slice_end      = number_channels - channel_offset;

                    int64_t spectrum_freq_slice_start = channel_offset;
                    int64_t spectrum_freq_slice_end   = number_channels;

                    auto drift_slice = bland::slice(drift_plane,
                                                    bland::slice_spec{0, drift_index, drift_index + 1},
                                                    bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});

                    fmt::print("Channel offset is {} so drift_slice[, {}:{}] + spectrum_slice[, {}:{}]\n", channel_offset, drift_freq_slice_start, drift_freq_slice_end, spectrum_freq_slice_start, spectrum_freq_slice_end);

                    auto spectrum_slice =
                            bland::slice(spectrum_grid,
                                        bland::slice_spec{0, t, t + 1},
                                        bland::slice_spec{1, spectrum_freq_slice_start, spectrum_freq_slice_end});

                    drift_slice = drift_slice + spectrum_slice / desmear_bandwidth;
                } else {
                    // At a negative drift rate, everything needs to scooch up instead of chop down
                    // the desmeared channel might need to be negative as well (it's the channel we're advancing towards)
                    auto channel_offset = freq_offset_at_time - desmear_channel;

                    int64_t drift_freq_slice_start    = -drift_channels;
                    int64_t drift_freq_slice_end      = number_channels;

                    int64_t spectrum_freq_slice_start = -drift_channels + channel_offset;
                    int64_t spectrum_freq_slice_end   = number_channels + channel_offset;

                    auto drift_slice = bland::slice(drift_plane,
                                                    bland::slice_spec{0, drift_index, drift_index + 1},
                                                    bland::slice_spec{1, drift_freq_slice_start, drift_freq_slice_end});
                    auto spectrum_slice =
                            bland::slice(spectrum_grid,
                                        bland::slice_spec{0, t, t + 1},
                                        bland::slice_spec{1, spectrum_freq_slice_start, spectrum_freq_slice_end});

                    drift_slice = drift_slice + spectrum_slice / desmear_bandwidth;

                }
            }

            // }
            // TODO: can do an in-place addition to remove ~50% of runtime
            // if (options.desmear && smeared_channels > 1) {
            //     for (int additional_channel=0; additional_channel < smeared_channels; ++additional_channel) {
            //         auto additional_drift_slice = bland::slice(drift_plane, bland::slice_spec{0, drift_step,
            //         drift_step+1}, bland::slice_spec{1, 0,
            //         drift_plane.size(1)-freq_offset_at_time-additional_channel}); auto additional_freq_slice =
            //         bland::slice(spectrum_grid, bland::slice_spec{0, t, t+1}, bland::slice_spec{1,
            //         freq_offset_at_time+additional_channel});

            //         additional_drift_slice = additional_drift_slice + additional_freq_slice/smeared_channels;
            //     }
            // }
        }
    }

    // normalize back by integration length
    return drift_plane / time_steps;
}

} // namespace detail

bland::ndarray bliss::integrate_drifts(const bland::ndarray &spectrum_grid, integrate_drifts_options options) {

    auto drift_grid = detail::integrate_linear_rounded_bins(spectrum_grid, options);

    return drift_grid;
}

doppler_spectrum bliss::integrate_drifts(filterbank_data fil_data, integrate_drifts_options options) {

    auto drift_grid = detail::integrate_linear_rounded_bins(fil_data.data(), options);

    return doppler_spectrum(fil_data, drift_grid, options);
}
