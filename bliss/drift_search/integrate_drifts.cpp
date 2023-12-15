
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

    int print_count = 0;
    // We use all time available inside this function
    for (int drift_step = options.low_rate; drift_step < options.high_rate; drift_step += options.rate_step_size) {

        auto m = static_cast<float>(drift_step) / static_cast<float>(maximum_drift_span);
        // fmt::print("Drift step {} translates to spectrum slope {}\n", drift_step, m);
        auto smeared_channels = std::round(std::abs(m));

        int number_integrated_channels = 1;
        if (options.desmear) {
            number_integrated_channels = std::max(1.0f, smeared_channels);
        }
        fmt::print("drift step {} (m={})has {} smeared channels, so {} number_integrated_channels\n",
                   drift_step,
                   m,
                   smeared_channels,
                   number_integrated_channels);
        for (int t = 0; t < time_steps; ++t) {
            int freq_offset_at_time = std::round(m * t);

            for (int channels_to_integrate = 0; channels_to_integrate < number_integrated_channels; ++channels_to_integrate) {
                auto drift_slice = bland::slice(
                        drift_plane,
                        bland::slice_spec{0, drift_step, drift_step + 1},
                        bland::slice_spec{1, 0, drift_plane.size(1) - freq_offset_at_time - channels_to_integrate});
                auto spectrum_slice = bland::slice(spectrum_grid,
                                                   bland::slice_spec{0, t, t + 1},
                                                   bland::slice_spec{1, freq_offset_at_time + channels_to_integrate});

                drift_slice = drift_slice + spectrum_slice / number_integrated_channels;
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
