
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

#include <chrono>
#include <iostream>
#include "clipp.h"

int main(int argc, char *argv[]) {

    std::vector<std::string> pipeline_files;
    int coarse_channel=0;
    int number_coarse=1;
    bliss::integrate_drifts_options dedrift_options{
            .desmear = true, .low_rate = -500, .high_rate = 500, .rate_step_size = 1};
    std::string device="cuda:0";
    int nchan_per_coarse=0;
    bliss::hit_search_options hit_search_options{.method = bliss::hit_search_methods::CONNECTED_COMPONENTS, .snr_threshold = 10.0f, .neighbor_l1_dist=7};
    bool help = false;
    auto cli = (
        (
            clipp::values("files").set(pipeline_files) % "input hdf5 filterbank files",
            (clipp::option("-c", "--coarse-channel") & clipp::value("coarse_channel").set(coarse_channel)) % "Coarse channel to process",
            (clipp::option("--number-coarse") & clipp::value("number_coarse").set(number_coarse)) % "Number of coarse channels to process",
            (clipp::option("-d", "--device") & clipp::value("device").set(device)) % "Compute device to use",
            (clipp::option("--desmear") .set(dedrift_options.desmear, true) |
             clipp::option("--nodesmear").set(dedrift_options.desmear, false)) % "Desmear the drift plane to compensate for drift rate crossing channels",
            (clipp::option("-m", "--min-rate") & clipp::value("min-rate").set(dedrift_options.low_rate)) % "Minimum drift rate (-5 Hz/sec)",
            (clipp::option("-M", "--max-rate") & clipp::value("max-rate").set(dedrift_options.high_rate)) % "Maximum drift rate (+5 Hz/sec)",
            (clipp::option("-s", "--snr") & clipp::value("snr_threshold").set(hit_search_options.snr_threshold)) % "SNR threshold (10)",
            (clipp::option("--nchan-per-coarse") & clipp::value("nchan_per_coarse").set(nchan_per_coarse)) % "number of fine channels per coarse to use (default: 0 auto-detects)"
        )
        |
        clipp::option("-h", "--help").set(help) % "Show this screen."
    );

    auto parse_result = clipp::parse(argc, argv, cli);

    if (!parse_result || help) {
        std::cout << clipp::make_man_page(cli, "bliss_find_hits");
        return 0;
    }

    auto pipeline_object = bliss::observation_target(pipeline_files, nchan_per_coarse);

    pipeline_object = pipeline_object.slice_observation_channels(coarse_channel, number_coarse);

    pipeline_object.set_device(device);

    pipeline_object = bliss::flag_filter_rolloff(pipeline_object, 0.2);
    pipeline_object = bliss::flag_spectral_kurtosis(pipeline_object, 0.1, 25);

    pipeline_object = bliss::estimate_noise_power(
            pipeline_object,
            bliss::noise_power_estimate_options{.estimator_method = bliss::noise_power_estimator::STDDEV,
                                                .masked_estimate  = true}); // estimate noise power of unflagged data

    pipeline_object = bliss::integrate_drifts(pipeline_object, dedrift_options);

    pipeline_object.set_device("cpu");

    auto pipeline_object_with_hits = bliss::hit_search(
            pipeline_object, hit_search_options);

    // TODO: add cli args for where to send hits (stdout, file.dat, capn proto serialize,...)
    for (auto &sc : pipeline_object_with_hits._scans) {
        auto hits = sc.hits();
        fmt::print("scan has {} hits\n", hits.size());
        for (auto &h : hits) {
            fmt::print("{}\n", h.repr());
        }
        bliss::write_hits_to_file(hits, "hitsout.cp");
    }

}
