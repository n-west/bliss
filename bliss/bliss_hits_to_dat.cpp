
#include <file_types/cpnp_files.hpp>
// #include <core/scan.hpp>
// #include <core/cadence.hpp>
// #include <estimators/noise_estimate.hpp>
// #include <drift_search/event_search.hpp>
// #include <drift_search/filter_hits.hpp>
// #include <drift_search/hit_search.hpp>
// #include <drift_search/integrate_drifts.hpp>
// #include <flaggers/filter_rolloff.hpp>
// #include <flaggers/magnitude.hpp>
// #include <flaggers/spectral_kurtosis.hpp>
// #include <file_types/hits_file.hpp>
// #include <file_types/events_file.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>
#include <cstdint>
#include <string>
#include <vector>

#include <chrono>
#include <iostream>
#include "clipp.h"

int main(int argc, char *argv[]) {

    std::vector<std::string> hit_files;
    bool help = false;
    auto cli = (
        (
            clipp::values("files").set(hit_files) % "input hdf5 filterbank files"
        )
        |
        clipp::option("-h", "--help").set(help) % "Show this screen."
    );

    auto parse_result = clipp::parse(argc, argv, cli);

    if (!parse_result || help) {
        std::cout << clipp::make_man_page(cli, "bliss_find_hits");
        return 0;
    }

    for (const auto &f : hit_files) {
        auto scan_with_hits = bliss::read_coarse_channel_hits_from_file(f);
        auto hits = scan_with_hits.hits();
        fmt::print("Got {} hits\n", hits.size());
        for (auto &h : hits) {
            std::cout << h.repr() << std::endl;
        }

    }

    // // TODO: add cli args for where to send hits (stdout, file.dat, capn proto serialize,...)
    // for (auto &sc : ) {
    //     auto hits = sc.hits();
    //     fmt::print("scan has {} hits\n", hits.size());
    //     for (auto &h : hits) {
    //         fmt::print("{}\n", h.repr());
    //     }
    // }

}
