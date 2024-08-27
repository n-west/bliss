
#include <core/scan.hpp>
#include <core/cadence.hpp>
#include <estimators/noise_estimate.hpp>
#include <preprocess/normalize.hpp>
#include <preprocess/passband_static_equalize.hpp>
#include <drift_search/event_search.hpp>
#include <drift_search/filter_hits.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/integrate_drifts.hpp>
#include <flaggers/filter_rolloff.hpp>
#include <flaggers/magnitude.hpp>
#include <flaggers/spectral_kurtosis.hpp>
#include <file_types/hits_file.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>
#include <cstdint>
#include <string>
#include <vector>

#include <chrono>
#include <iostream> // for printing help
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "Filesystem library not available"
#endif

#include "clipp.h"

int main(int argc, char *argv[]) {

    std::vector<std::string> pipeline_files;
    int coarse_channel=0;
    int number_coarse_channels=1;
    std::string channel_taps_path;
    bliss::integrate_drifts_options dedrift_options{
            .desmear = true, .low_rate = -500, .high_rate = 500, .rate_step_size = 1};
    std::string device="cuda:0";
    int nchan_per_coarse=0;
    bliss::hit_search_options hit_search_options{.method = bliss::hit_search_methods::CONNECTED_COMPONENTS, .snr_threshold = 10.0f, .neighbor_l1_dist=7};
    std::string output_path = "";
    std::string output_format = "";
    bool help = false;
    auto cli = (
        (
            clipp::values("files").set(pipeline_files) % "input hdf5 filterbank files",

            // Data slicing before any processing
            (clipp::option("-c", "--coarse-channel") & clipp::value("coarse_channel").set(coarse_channel)) % "Coarse channel to process",
            (clipp::option("--number-coarse") & clipp::value("number_coarse_channels").set(number_coarse_channels)) % "Number of coarse channels to process (default: 1)",
            (clipp::option("--nchan-per-coarse") & clipp::value("nchan_per_coarse").set(nchan_per_coarse)) % "number of fine channels per coarse to use (default: 0 auto-detects)",

            // Preprocessing
            (clipp::option("-e", "--equalizer-channel") & clipp::value("channel_taps").set(channel_taps_path)) % "the path to coarse channel response at fine frequency resolution",

            // Compute device / params
            (clipp::option("-d", "--device") & clipp::value("device").set(device)) % "Compute device to use",

            // Drift intgration / dedoppler
            (clipp::option("--desmear") .set(dedrift_options.desmear, true) |
             clipp::option("--nodesmear").set(dedrift_options.desmear, false)) % "Desmear the drift plane to compensate for drift rate crossing channels",
            (clipp::option("-m", "--min-rate") & clipp::value("min-rate").set(dedrift_options.low_rate)) % "Minimum drift rate (-5 Hz/sec)",
            (clipp::option("-M", "--max-rate") & clipp::value("max-rate").set(dedrift_options.high_rate)) % "Maximum drift rate (+5 Hz/sec)",

            // Hit search
            (clipp::option("--local-maxima") .set(hit_search_options.method, bliss::hit_search_methods::LOCAL_MAXIMA) |
             clipp::option("--connected-components").set(hit_search_options.method, bliss::hit_search_methods::CONNECTED_COMPONENTS)) % "select the hit search method",
            (clipp::option("-s", "--snr") & clipp::value("snr_threshold").set(hit_search_options.snr_threshold)) % "SNR threshold (10)",
            (clipp::option("--distance") & clipp::value("l1_distance").set(hit_search_options.neighbor_l1_dist)) % "L1 distance to consider hits connected (7)",

            (clipp::option("-o", "--output") & (clipp::value("output_file").set(output_path), clipp::opt_value("format").set(output_format))) % "Filename to store output"
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

    pipeline_object = pipeline_object.slice_observation_channels(coarse_channel, number_coarse_channels);

    pipeline_object.set_device(device);

    pipeline_object = bliss::normalize(pipeline_object);
    if (!channel_taps_path.empty()) {
        pipeline_object = bliss::equalize_passband_filter(pipeline_object, channel_taps_path);
    }

    pipeline_object = bliss::flag_filter_rolloff(pipeline_object, 0.25);
    pipeline_object = bliss::flag_spectral_kurtosis(pipeline_object, 0.1, 25);

    pipeline_object = bliss::estimate_noise_power(
            pipeline_object,
            bliss::noise_power_estimate_options{.estimator_method = bliss::noise_power_estimator::STDDEV,
                                                .masked_estimate  = true}); // estimate noise power of unflagged data

    pipeline_object = bliss::integrate_drifts(pipeline_object, dedrift_options);

    auto pipeline_object_with_hits = bliss::hit_search(pipeline_object, hit_search_options);

    pipeline_object_with_hits = bliss::filter_hits(pipeline_object_with_hits, bliss::filter_options{.filter_zero_drift = true});

    try {
        // TODO: add cli args for where to send hits (stdout, file.dat, capn proto serialize,...)
        for (int scan_index=0; scan_index < pipeline_object_with_hits._scans.size(); ++scan_index) {
            auto &sc = pipeline_object_with_hits._scans[scan_index];
            auto hits = sc.hits();

            if (output_path.empty()) {
                auto path = fs::path(pipeline_files[scan_index]);

                output_path = path.filename().replace_extension("capnp");
                output_format = "capnp";
            }

            if (output_path == "-" || output_path == "stdout") {
                fmt::print("scan has {} hits\n", hits.size());
                for (auto &h : hits) {
                    fmt::print("{}\n", h.repr());
                }
            } else {
                bliss::write_scan_hits_to_file(sc, output_path, output_format);
            }
        }
    } catch (std::exception &e) {
        fmt::print("ERROR: got a fatal exception ({}) while running pipeline. This is likely due to running out of "
                   "memory. Ending processing.\n",
                   e.what());
    }

}
