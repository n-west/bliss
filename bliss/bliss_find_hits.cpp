
#include <core/scan.hpp>
#include <core/cadence.hpp>
#include <estimators/noise_estimate.hpp>
#include <preprocess/excise_dc.hpp>
#include <preprocess/normalize.hpp>
#include <preprocess/passband_static_equalize.hpp>
#include <drift_search/event_search.hpp>
#include <drift_search/filter_hits.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/integrate_drifts.hpp>
#include <flaggers/filter_rolloff.hpp>
#include <flaggers/magnitude.hpp>
#include <flaggers/sigmaclip.hpp>
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
    bool validate_pfb_response = false;
    bool excise_dc = false;
    bliss::integrate_drifts_options dedrift_options{
            .desmear = true, .low_rate_Hz_per_sec = -5, .high_rate_Hz_per_sec = 5, .resolution = 1};
    int low_rate = std::numeric_limits<int>::min();
    int high_rate = std::numeric_limits<int>::max();

    std::string device="cuda:0";
    int nchan_per_coarse=0;
    bliss::hit_search_options hit_search_options{.method = bliss::hit_search_methods::CONNECTED_COMPONENTS, .snr_threshold = 10.0f, .neighbor_l1_dist=7};
    bliss::filter_options hit_filter_options{.filter_zero_drift = true,
                .filter_sigmaclip = false, .minimum_percent_sigmaclip = 0.1,
                .filter_high_sk = false, .minimum_percent_high_sk = 0.1,
                .filter_low_sk = false, .maximum_percent_low_sk = 0.1};
    std::string output_path = "";
    std::string output_format = "";
    struct {
        float filter_rolloff = 0.25;
        float sk_low = 0.25;
        float sk_high = 50;
        float sk_d = 2;
        int sigmaclip_iters = 5;
        float sigmaclip_low = 3;
        float sigmaclip_high = 4;
    } flag_options;
    bool help = false;
    auto cli = (
        (
            clipp::values("files").set(pipeline_files) % "input hdf5 filterbank files",

            // Data slicing before any processing
            (clipp::option("-c", "--coarse-channel") & clipp::value("coarse_channel").set(coarse_channel)) % fmt::format("Coarse channel to process (default: {}])", coarse_channel),
            (clipp::option("--number-coarse") & clipp::value("number_coarse_channels").set(number_coarse_channels)) % fmt::format("Number of coarse channels to process (default: {})", number_coarse_channels),
            (clipp::option("--nchan-per-coarse") & clipp::value("nchan_per_coarse").set(nchan_per_coarse)) % fmt::format("number of fine channels per coarse to use (default: {} auto-detects)", nchan_per_coarse),

            // Preprocessing
            (clipp::option("-e", "--equalizer-channel") & clipp::value("channel_taps").set(channel_taps_path)) % "the path to coarse channel response at fine frequency resolution",
            (clipp::option("--validate-pfb") & clipp::value("channel_taps").set(validate_pfb_response)) % fmt::format("whether to validate the coarse channel has a similar PFB response to the given response (default: {})", validate_pfb_response),
            (clipp::option("--excise-dc") .set(dedrift_options.desmear, true) |
             clipp::option("--noexcise-dc").set(dedrift_options.desmear, false)) % fmt::format("Excise DC offset from the data (default: {})", excise_dc),

            // Compute device / params
            (clipp::option("-d", "--device") & clipp::value("device").set(device)) % fmt::format("Compute device to use (default: {})", device),

            // Drift intgration / dedoppler
            (clipp::option("--desmear") .set(dedrift_options.desmear, true) |
             clipp::option("--nodesmear").set(dedrift_options.desmear, false)) % "Desmear the drift plane to compensate for drift rate crossing channels",
            (clipp::option("-md", "--min-drift") & clipp::value("min-rate").set(dedrift_options.low_rate_Hz_per_sec)) % fmt::format("Minimum drift rate (default: {})", dedrift_options.low_rate_Hz_per_sec),
            (clipp::option("-MD", "--max-drift") & clipp::value("max-rate").set(dedrift_options.high_rate_Hz_per_sec)) % fmt::format("Maximum drift rate (default: {})", dedrift_options.high_rate_Hz_per_sec),
            // Reserve for potential Hz/sec step in the future
            // (clipp::option("-dr", "--drift-resolution") & clipp::value("rate-step").set(dedrift_options.resolution)) % "Multiple of unit drift resolution to step in search (default: 1)",
            (clipp::option("-rs", "--rate-step") & clipp::value("rate-step").set(dedrift_options.resolution)) % fmt::format("Multiple of unit drift resolution to step in search (default: {})", dedrift_options.resolution),
            
            (clipp::option("-m", "--min-rate") & clipp::value("min-rate").set(low_rate)) % "(DEPRECATED: use -md) Minimum drift rate (fourier bins)",
            (clipp::option("-M", "--max-rate") & clipp::value("max-rate").set(high_rate)) % "(DEPRECATED: use -MD) Maximum drift rate (fourier bins)",

            // Flagging
            // This is left in as a reference for refactoring and cleaning up cli in the future. I would like to have these subgroups, etc but
            // don't want to break things that people are already used to. I'll make a big announcement about the breaking change when I do
            // (clipp::option("--flag").doc("Flagging options") & (
            //     // Filter rolloff
            //     (clipp::option("--filter-rolloff").doc("Enable filter rolloff flagging") & 
            //         clipp::opt_value("fraction").set(flag_options.filter_rolloff)) % fmt::format("Fraction of band edges to flag (default: {})", flag_options.filter_rolloff),
                
            //     // Sigma clipping
            //     (clipp::option("--sigmaclip").doc(fmt::format("Set the iterations for sigma clipping (default: {})", flag_options.sigmaclip_iters)) & (
            //         clipp::opt_value("iters").set(flag_options.sigmaclip_iters),
            //         (clipp::option("--low").doc("Set the lower threshold for sigma clipping") & clipp::value("threshold").set(flag_options.sigmaclip_low)),
            //         (clipp::option("--high").doc("Set the upper threshold for sigma clipping") & clipp::value("threshold").set(flag_options.sigmaclip_high))
            //     )) % fmt::format("Sigma clipping with iterations={}, low={}, high={}", 
            //         flag_options.sigmaclip_iters, flag_options.sigmaclip_low, flag_options.sigmaclip_high),
                
            //     // Spectral kurtosis
            //     (clipp::option("--sk").doc("Enable spectral kurtosis flagging") & (
            //         (clipp::option("--low").doc("Set the lower threshold for spectral kurtosis") & clipp::value("threshold").set(flag_options.sk_low)),
            //         (clipp::option("--high").doc("Set the upper threshold for spectral kurtosis") & clipp::value("threshold").set(flag_options.sk_high)),
            //         (clipp::option("-d", "--d").doc("Set the shape parameter for spectral kurtosis") & clipp::value("shape").set(flag_options.sk_d))
            //     )) % fmt::format("Spectral kurtosis with low={}, high={}, d={}", 
            //         flag_options.sk_low, flag_options.sk_high, flag_options.sk_d)
            // )),

            // (clipp::option("--flag").doc("Flagging options") & (
            //     // Filter rolloff
            //     (clipp::option("--filter-rolloff").doc("Enable filter rolloff flagging") & 
            //         clipp::opt_value("fraction").set(flag_options.filter_rolloff)) % fmt::format("Fraction of band edges to flag (default: {})", flag_options.filter_rolloff),
                
            //     // Sigma clipping
            //     (clipp::option("--sigmaclip").doc(fmt::format("Set the iterations for sigma clipping (default: {})", flag_options.sigmaclip_iters)) & (
            //         clipp::opt_value("iters").set(flag_options.sigmaclip_iters),
            //         (clipp::option("--low").doc("Set the lower threshold for sigma clipping") & clipp::value("threshold").set(flag_options.sigmaclip_low)),
            //         (clipp::option("--high").doc("Set the upper threshold for sigma clipping") & clipp::value("threshold").set(flag_options.sigmaclip_high))
            //     )) % fmt::format("Sigma clipping with iterations={}, low={}, high={}", 
            //         flag_options.sigmaclip_iters, flag_options.sigmaclip_low, flag_options.sigmaclip_high),
                
            (clipp::option("--sk-low") & clipp::value("spectral kurtosis lower").set(flag_options.sk_low)) % "Flagging lower threshold for spectral kurtosis",
            (clipp::option("--sk-high") & clipp::value("spectral kurtsosis high").set(flag_options.sk_high)) % "Flagging high threshold for spectral kurtosis",
            (clipp::option("--sk-d") & clipp::value("spectral kurtosis d").set(flag_options.sk_d)) % "Flagging shape parameter for spectral kurtosis",

            // Hit search
            (clipp::option("--local-maxima") .set(hit_search_options.method, bliss::hit_search_methods::LOCAL_MAXIMA) |
             clipp::option("--connected-components").set(hit_search_options.method, bliss::hit_search_methods::CONNECTED_COMPONENTS)) % "select the hit search method",
            (clipp::option("-s", "--snr") & clipp::value("snr_threshold").set(hit_search_options.snr_threshold)) % fmt::format("SNR threshold (default: {})", hit_search_options.snr_threshold),
            (clipp::option("--distance") & clipp::value("l1_distance").set(hit_search_options.neighbor_l1_dist)) % fmt::format("L1 distance to consider hits connected (default: {})", hit_search_options.neighbor_l1_dist),

            // Hit filtering
            (clipp::option("--filter-zero-drift") .set(hit_filter_options.filter_zero_drift, true) |
             clipp::option("--nofilter-zero-drift").set(hit_filter_options.filter_zero_drift, false)) % fmt::format("Filter out hits with zero drift rate (default: {})", hit_filter_options.filter_zero_drift),

            (clipp::option("--filter-sigmaclip") .set(hit_filter_options.filter_sigmaclip, true) |
             clipp::option("--nofilter-sigmaclip").set(hit_filter_options.filter_sigmaclip, false)) % fmt::format("Filter out hits with low sigmaclip (default: {})", hit_filter_options.filter_sigmaclip),
            (clipp::option("--min-sigmaclip") & clipp::value("min_sigmaclip").set(hit_filter_options.minimum_percent_sigmaclip)) % fmt::format("Minimum percent of sigmaclip to keep (default: {})", hit_filter_options.minimum_percent_sigmaclip),

            (clipp::option("--filter-high-sk") .set(hit_filter_options.filter_high_sk, true) |
             clipp::option("--nofilter-high-sk").set(hit_filter_options.filter_high_sk, false)) % fmt::format("Filter out hits with high spectral kurtosis (default: {})", hit_filter_options.filter_high_sk),
            (clipp::option("--min-high-sk") & clipp::value("min_high_sk").set(hit_filter_options.minimum_percent_high_sk)) % fmt::format("Minimum percent of high spectral kurtosis to keep (default: {})", hit_filter_options.minimum_percent_high_sk),

            (clipp::option("--filter-low-sk") .set(hit_filter_options.filter_low_sk, true) |
             clipp::option("--nofilter-low-sk").set(hit_filter_options.filter_low_sk, false)) % fmt::format("Filter out hits with low spectral kurtosis (default: {})", hit_filter_options.filter_low_sk),
            (clipp::option("--max-low-sk") & clipp::value("max_low_sk").set(hit_filter_options.maximum_percent_low_sk)) % fmt::format("Maximum percent of low spectral kurtosis to keep (default: {})", hit_filter_options.maximum_percent_low_sk),

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


    auto foff = std::fabs(1e6 * pipeline_object._scans[0].foff());
    auto tsamp = pipeline_object._scans[0].tsamp();
    auto ntsteps = pipeline_object._scans[0].ntsteps();
    auto drift_resolution = foff/(tsamp*(ntsteps-1));
    if (low_rate != std::numeric_limits<int>::min()) {
        auto low_rate_Hz_per_sec = low_rate * drift_resolution;
        dedrift_options.low_rate_Hz_per_sec = low_rate_Hz_per_sec;
        fmt::print("WARN: deprecated use of -m (value given is {}) to specify min drift rate in terms of unit drift resolution bins. Use -md {} to specify min drift in units of Hz/sec instead.\n", low_rate, low_rate_Hz_per_sec);
    }
    if (high_rate != std::numeric_limits<int>::max()) {
        auto high_rate_Hz_per_sec = high_rate * drift_resolution;        
        dedrift_options.high_rate_Hz_per_sec = high_rate_Hz_per_sec;
        fmt::print("WARN: deprecated use of -M (value given is {}) to specify Max drift rate in terms of unit drift resolution bins. Use -MD {} to specify Max Drift in units of Hz/sec instead.\n", high_rate, high_rate_Hz_per_sec);
    }


    pipeline_object.set_device(device);

    pipeline_object = bliss::normalize(pipeline_object);
    pipeline_object = bliss::excise_dc(pipeline_object);
    if (!channel_taps_path.empty()) {
        pipeline_object = bliss::equalize_passband_filter(pipeline_object, channel_taps_path, bland::ndarray::datatype::float32, true);
    } else {
        pipeline_object = bliss::flag_filter_rolloff(pipeline_object, flag_options.filter_rolloff);
    }
    pipeline_object = bliss::flag_spectral_kurtosis(pipeline_object, flag_options.sk_low, flag_options.sk_high, flag_options.sk_d);
    pipeline_object = bliss::flag_sigmaclip(pipeline_object, flag_options.sigmaclip_iters, flag_options.sigmaclip_low, flag_options.sigmaclip_high);

    pipeline_object = bliss::estimate_noise_power(
            pipeline_object,
            bliss::noise_power_estimate_options{.estimator_method = bliss::noise_power_estimator::STDDEV,
                                                .masked_estimate  = true}); // estimate noise power of unflagged data

    pipeline_object = bliss::integrate_drifts(pipeline_object, dedrift_options);

    auto pipeline_object_with_hits = bliss::hit_search(pipeline_object, hit_search_options);

    pipeline_object_with_hits = bliss::filter_hits(pipeline_object_with_hits, hit_filter_options);

    try {
        // TODO: add cli args for where to send hits (stdout, file.dat, capn proto serialize,...)
        for (int scan_index=0; scan_index < pipeline_object_with_hits._scans.size(); ++scan_index) {
            auto &sc = pipeline_object_with_hits._scans[scan_index];

            if (output_path.empty()) {
                auto path = fs::path(pipeline_files[scan_index]);

                output_path = path.filename().replace_extension("capnp");
                output_format = "capnp";
            }

            if (output_path == "-" || output_path == "stdout") {
                auto hits = sc.hits();
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
