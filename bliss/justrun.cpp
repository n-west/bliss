
#include <core/filterbank_data.hpp>
#include <flaggers/filter_rolloff.hpp>
#include <flaggers/magnitude.hpp>
#include <flaggers/spectral_kurtosis.hpp>
#include <estimators/noise_estimate.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/integrate_drifts.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <vector>

int main() {

    auto fil_data = bliss::filterbank_data("/home/nathan/datasets/voyager_2020_data/"
                                           "single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5");

    // TODO: add support for vector of looks (a cadence dtype that holds all dwells)

    auto flagged_fil = bliss::flag_spectral_kurtosis(fil_data, 0.01f, 10.0f); // SK > 10 && SK < .01
    // auto flagged_fil = bliss::flag_magnitude(flagged_fil, 10.0f); // mag > mean + std*10
    flagged_fil = bliss::flag_filter_rolloff(flagged_fil, .05f); // % of band edges
    // auto flagged_fil = bliss::flag_ood(flagged_fil, .01); // < 1% change belonging to predicted noise only
    // distribution

    auto noise_stats = bliss::estimate_noise_power(fil_data, bliss::noise_power_estimate_options{.masked_estimate=true}); // estimate noise power of unflagged data

    auto dedrifted_fil = bliss::integrate_drifts(flagged_fil,
                                                bliss::integrate_drifts_options{.desmear = true}); // integrate along drift lines

    // auto hits = bliss::hit_search(dedrifted_fil, noise_stats, 10.0f);

    // bliss::write_hits(hits);

}
