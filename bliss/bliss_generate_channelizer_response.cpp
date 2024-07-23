
#include <core/scan.hpp>
#include <core/cadence.hpp>

#include <preprocess/passband_static_equalize.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>
#include <cstdint>
#include <string>
#include <vector>

#include <chrono>
#include <iostream>
#include "clipp.h"

int main(int argc, char *argv[]) {

    std::string out_file = "channelizer_response.f32";
    int number_coarse_channels=192;
    std::string device="cpu";
    int nchan_per_coarse = 131072;
    int taps_per_channel = 12;
    bool help = false;
    auto cli = (
        (
            // Filter design specs
            (clipp::option("-M", "--number-coarse") & clipp::value("number_coarse_channels").set(number_coarse_channels)) % "Number of coarse channels to process (default: 192)",
            (clipp::option("-f", "--nchan-per-coarse") & clipp::value("nchan_per_coarse").set(nchan_per_coarse)) % "number of fine channels per coarse to use (default: 2**20)",
            (clipp::option("-N", "--taps-per-channel") & clipp::value("taps_per_channel").set(taps_per_channel)) % "number of fine channels per coarse to use (default: 12)",

            // Compute device / params
            (clipp::option("-d", "--device") & clipp::value("device").set(device)) % "Compute device to use",

            // Output params
            (clipp::option("-o", "--output") & clipp::value("output_file").set(out_file)) % "file to write response to (default: channelizer_response.f32)"
        )
        |
        clipp::option("-h", "--help").set(help) % "Show this screen."
    );

    auto parse_result = clipp::parse(argc, argv, cli);

    if (!parse_result || help) {
        auto man_page = clipp::make_man_page(cli, "bliss_generate_channelizer_response");
        man_page = man_page.append_section("EXAMPLES",
        "ATA: ./bliss_generate_channelizer_response -f 131072 -N 4 -M 2048\n"
        "GBT: ./bliss_generate_channelizer_response -f 1048576 -N 12 -M 256\n");

        std::cout << man_page;
        return 0;
    }

    // auto h_resp = bliss::gen_coarse_channel_response(131072, 2048, 4);
    auto h_resp = bliss::gen_coarse_channel_response(nchan_per_coarse, number_coarse_channels, taps_per_channel);

    bland::write_to_file(h_resp, out_file);
}
