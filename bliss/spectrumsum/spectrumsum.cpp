
#include "spectrumsum/spectrumsum.hpp"

#include <bland/bland.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

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
[[nodiscard]] bland::ndarray spectrum_sum_linear_round(const bland::ndarray &spectrum_grid, spectrum_sum_options options) {
    auto number_drifts = options.drift_range;
    fmt::print("*** BLISS ***: In spectrum sum w/ LINEAR_ROUND\n");

    auto time_steps = spectrum_grid.size(0);
    auto number_channels = spectrum_grid.size(1);
    // number_drifts = time_steps;

    auto maximum_drift_span = time_steps - 1;


    bland::ndarray drift_plane({number_drifts, number_channels}, spectrum_grid.dtype(), spectrum_grid.device());
    bland::fill(drift_plane, 0.0f);

    int print_count = 0;
    fmt::print("Input spectrum grid has shape {}, detection plane has shape {}\n", spectrum_grid.shape(), drift_plane.shape());
    // We use all time available inside this function
    for (int drift_step = 0; drift_step < number_drifts; ++drift_step) {

        auto m = static_cast<float>(drift_step) / static_cast<float>(maximum_drift_span);
        // fmt::print("Drift step {} translates to spectrum slope {}\n", drift_step, m);
        auto smeared_channels = std::round(m);

        int number_integrated_channels = 1;
        if (options.desmear) {
            number_integrated_channels = std::max(1.0f, smeared_channels);
        }
        fmt::print("drift step {} (m={})has {} smeared channels, so {} number_integrated_channels\n", drift_step, m, smeared_channels, number_integrated_channels);
        for (int t = 0; t < time_steps; ++t) {
            int freq_offset_at_time = std::round(m * t);

            for (int channels_to_integrate = 0; channels_to_integrate < number_integrated_channels; ++channels_to_integrate) {
                    auto drift_slice = bland::slice(drift_plane, bland::slice_spec{0, drift_step, drift_step+1}, bland::slice_spec{1, 0, drift_plane.size(1)-freq_offset_at_time-channels_to_integrate});
                    auto spectrum_slice = bland::slice(spectrum_grid, bland::slice_spec{0, t, t+1}, bland::slice_spec{1, freq_offset_at_time+channels_to_integrate});

                    drift_slice = drift_slice + spectrum_slice/number_integrated_channels;
            }

            // }
            // TODO: can do an in-place addition to remove ~50% of runtime
            // if (options.desmear && smeared_channels > 1) {
            //     for (int additional_channel=0; additional_channel < smeared_channels; ++additional_channel) {
            //         auto additional_drift_slice = bland::slice(drift_plane, bland::slice_spec{0, drift_step, drift_step+1}, bland::slice_spec{1, 0, drift_plane.size(1)-freq_offset_at_time-additional_channel});
            //         auto additional_freq_slice = bland::slice(spectrum_grid, bland::slice_spec{0, t, t+1}, bland::slice_spec{1, freq_offset_at_time+additional_channel});

            //         additional_drift_slice = additional_drift_slice + additional_freq_slice/smeared_channels;
            //     }
            // }
        }
    }

    return drift_plane;
}

bland::ndarray spectrum_sum_taylor_tree(const bland::ndarray &spectrum_grid) {
    // std::cout << "Doing taylor tree path through spectrum" << std::endl;
    auto time_steps = spectrum_grid.size(0);

    auto maximum_drift_span = time_steps - 1;

    bland::ndarray drift_plane(spectrum_grid.shape(), spectrum_grid.dtype(), spectrum_grid.device());
    bland::fill(drift_plane, 0.0f);


    return drift_plane;
}

bland::ndarray spectrum_sum_houston(const bland::ndarray &spectrum_grid) {
    // TODO: Need to spend some more time deciphering this or talk to Ken
    // std::cout << "Doing houston-rounded path through spectrum" << std::endl;
    auto time_steps = spectrum_grid.size(0);

    auto maximum_drift_span = time_steps - 1;

    bland::ndarray drift_plane(spectrum_grid.shape(), spectrum_grid.dtype(), spectrum_grid.device());
    bland::fill(drift_plane, 0.0f);

    return drift_plane;
}
} // namespace detail

// [[nodiscard]] bland::ndarray spectrum_sum(const bland::ndarray &spectrum_grid,
//                                           spectrum_sum_options  options = spectrum_sum_options{
//                                                    .desmear = true,
//                                                    .method  = spectrum_sum_method::LINEAR_ROUND});
bland::ndarray bliss::spectrum_sum(const bland::ndarray &spectrum_grid, spectrum_sum_options options) {
    fmt::print("*** BLISS ***: In spectrum sum ({})\n", spectrum_grid.data_ptr<void>());

    // The default in most other incoherent energy spectrum sums has been taylor
    // tree-based so we will use that as the default method here as well.
    bland::ndarray drift_grid({1});
    switch (options.method) {
    case spectrum_sum_method::LINEAR_ROUND: {
        drift_grid = detail::spectrum_sum_linear_round(spectrum_grid, options);
    } break;
    case spectrum_sum_method::HOUSTON: {
        drift_grid = detail::spectrum_sum_houston(spectrum_grid);
    } break;
    case spectrum_sum_method::TAYLOR_TREE:
    default: {
        drift_grid = detail::spectrum_sum_taylor_tree(spectrum_grid);
    }
    }

    return drift_grid;
}
