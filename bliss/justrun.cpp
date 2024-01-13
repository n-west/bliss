
#include <core/filterbank_data.hpp>
#include <core/cadence.hpp>
#include <estimators/noise_estimate.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/integrate_drifts.hpp>
#include <flaggers/filter_rolloff.hpp>
#include <flaggers/magnitude.hpp>
#include <flaggers/spectral_kurtosis.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <vector>

int main(int argc, char **argv) {

    std::string fil_path{"/home/nathan/datasets/voyager_2020_data/"
                         "single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"};
    if (argc == 2) {
        fil_path = argv[1];
    }

    // bliss::cadence({{"/home/nathan/datasets/voyager_2020_data/"
    //                      "single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"}});
    auto cadence = bliss::cadence({{"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5",
                    "/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5",
                    "/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5"
                    },
                    {"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5"},
                    {"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5"},
                    {"/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5"}});

    // // Kinda weird, but the scoping is annoying and we don't have default constructors that make sense
    // std::unique_ptr<bliss::filterbank_data> fil_holder;
    // try {
    //   fil_holder = std::make_unique<bliss::filterbank_data>(fil_path);
    // } catch (std::exception) {
    //   fmt::print("Bad file path provided. Couldn't open {} as hdf5 filterbank file\n", fil_path);
    //   return -1;
    // }
    // auto fil_data = *fil_holder;

    cadence = bliss::flag_filter_rolloff(cadence, 0.2);
    cadence = bliss::flag_spectral_kurtosis(cadence, 0.02, 5);

    auto noise_stats = bliss::estimate_noise_power(
            cadence,
            bliss::noise_power_estimate_options{.masked_estimate = true}); // estimate noise power of unflagged data

    // auto dedrifted_fil = bliss::integrate_drifts(
    //         flagged_fil,
    //         bliss::integrate_drifts_options{.desmear        = true,
    //                                         .low_rate       = -8,
    //                                         .high_rate      = 48,
    //                                         .rate_step_size = 1}); // integrate along drift lines

    // auto hits = bliss::hit_search(dedrifted_fil, {.snr_threshold=10.0f});

    // bliss::write_hits(hits);
}
