
#include <core/scan.hpp>
#include <core/cadence.hpp>
#include <estimators/noise_estimate.hpp>
#include <drift_search/event_search.hpp>
#include <drift_search/filter_hits.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/integrate_drifts.hpp>
#include <flaggers/filter_rolloff.hpp>
#include <flaggers/magnitude.hpp>
#include <flaggers/spectral_kurtosis.hpp>
#include <file_types/hits_file.hpp>
#include <file_types/events_file.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <vector>

int main(int argc, char **argv) {


    // if (argc == 2) {
    //     fil_path = argv[1];
    // }

    // auto mlc4_cadence = bliss::cadence({{"/datag/public/seti_benchmarking/mlc4/spliced_blc0001020304050607_guppi_57517_08789_HIP54677_0009.gpuspec.0000.h5",
    //                  "/datag/public/seti_benchmarking/mlc4/spliced_blc0001020304050607_guppi_57517_09628_HIP54677_0011.gpuspec.0000.h5",
    //                  "/datag/public/seti_benchmarking/mlc4/spliced_blc0001020304050607_guppi_57517_10436_HIP54677_0013.gpuspec.0000.h5"},
    //                  {"/datag/public/seti_benchmarking/mlc4/spliced_blc0001020304050607_guppi_57517_09209_HIP53759_0010.gpuspec.0000.h5"},
    //                  {"/datag/public/seti_benchmarking/mlc4/spliced_blc0001020304050607_guppi_57517_10032_HIP53820_0012.gpuspec.0000.h5"},
    //                  {"/datag/public/seti_benchmarking/mlc4/spliced_blc0001020304050607_guppi_57517_10836_HIP53839_0014.gpuspec.0000.h5"}});


    auto voyager_cadence = bliss::cadence({{"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5",
                    "/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5",
                    "/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5"
                    },
                    {"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"},
                    {"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5"},
                    {"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5"}});

    // auto cadence = voyager_cadence.slice_cadence_channels(188);
    auto cadence = voyager_cadence;

    cadence = bliss::flag_filter_rolloff(cadence, 0.2);
    cadence = bliss::flag_spectral_kurtosis(cadence, 0.02, 25);

    cadence = bliss::estimate_noise_power(
            cadence,
            bliss::noise_power_estimate_options{.estimator_method=bliss::noise_power_estimator::STDDEV, .masked_estimate = true}); // estimate noise power of unflagged data

    cadence = bliss::integrate_drifts(
            cadence,
            bliss::integrate_drifts_options{.desmear        = false,
                                            .low_rate       = -48,
                                            .high_rate      = 48,
                                            .rate_step_size = 1});

    cadence = bliss::hit_search(cadence, {.method=bliss::hit_search_methods::CONNECTED_COMPONENTS, .snr_threshold=10.0f});

    bliss::write_cadence_hits_to_files(cadence, "hits");

    auto events = bliss::event_search(cadence);

    // cadence = bliss::filter_hits(cadence, {});

    bliss::write_events_to_file(events, "events_output");

}
