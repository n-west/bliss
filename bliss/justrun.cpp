
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

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

int main(int argc, char **argv) {

    std::string fil_path = "/datax/scratch/nwest/data/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5";
    // std::string fil_path = "/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5";
    if (argc == 2) {
        fil_path = argv[1];
    }

    auto voyager_cadence = bliss::scan("/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5");

    // auto voyager_cadence = bliss::cadence({{"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5",
    //                 "/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5",
    //                 "/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5"
    //                 },
    //                 {"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"},
    //                 {"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5"},
    //                 {"/datag/public/voyager_2020/single_coarse_channel/old_single_coarse/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5"}});

    auto cadence = voyager_cadence;

    cadence.set_device("cuda:0");
    // cadence.push_device();

    cadence = bliss::flag_filter_rolloff(cadence, 0.2);
    cadence = bliss::flag_spectral_kurtosis(cadence, 0.1, 25);

    cadence = bliss::estimate_noise_power(
            cadence,
            bliss::noise_power_estimate_options{.estimator_method=bliss::noise_power_estimator::STDDEV, .masked_estimate = true}); // estimate noise power of unflagged data

    cadence = bliss::integrate_drifts(
            cadence,
            bliss::integrate_drifts_options{.desmear        = true,
                                            .low_rate       = -500,
                                            .high_rate      = 500,
                                            .rate_step_size = 1});
    // cadence.set_device("cpu");

    auto cadence_with_hits = bliss::hit_search(cadence, {.method=bliss::hit_search_methods::CONNECTED_COMPONENTS,
                                                        .snr_threshold=10.0f});

    cadence_with_hits.set_device("cpu");

    auto hits = cadence_with_hits.hits();

    fmt::print("justrun: there are {} hits\n", hits.size());
    for (auto & h : hits) {
        fmt::print("{}\n", h.repr());
    }
    // auto events = bliss::event_search(cadence);

    // bliss::write_events_to_file(events, "events_output");

}
