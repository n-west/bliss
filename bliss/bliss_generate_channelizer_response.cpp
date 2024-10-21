
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
    int number_coarse_channels=256;
    std::string device="cpu";
    int nchan_per_coarse = 1048576;
    int taps_per_channel = 12;
    bool help = false;
    auto cli = (
        (
            // Filter design specs
            (clipp::option("-M", "--number-coarse") & clipp::value("number_coarse_channels").set(number_coarse_channels)) % "Number of coarse channels to process (default: 256)",
            (clipp::option("-f", "--nchan-per-coarse") & clipp::value("nchan_per_coarse").set(nchan_per_coarse)) % "number of fine channels per coarse to use (default: 2**20)",
            (clipp::option("-N", "--taps-per-channel") & clipp::value("taps_per_channel").set(taps_per_channel)) % "number of taps per coarse channel (default: 12)",

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
        man_page = man_page.append_section("SYNOPSIS",
        "Most radio telescopes with a SETI/technosignature backend use a polyphase filterbank (pfb) configured to "
        "channelize sampled spectra in to 'coarse channels' before generating some number of 'fine channels' "
        "used for signal searches. The resulting fine channels are shaped by a combination of actual spectral  "
        "data in sampled bandwidth and the amplitude response of the first pfb's taps. This tool will generate "
        "a 32-bit floating-point array of the amplitude response of those coarse channels sampled with the frequency "
        "resolution of fine channels.\n"
        "\n"
        "The process works as follows:\n"
        "1) recreate the taps used for the coarse channel pfb assuming a hamming window was used during window method of filter design. This requires knowing the number of taps per coarse channel. This array is often called the prototype filter taps in literature and is the number of taps per coarse channel * number of coarse channels long.\n"
        "2) zero-pad the prototype filter taps to be number of coarse channels * number of fine channels per coarse long. This is so the FFT in the next step is sampled with the frequency resolution of fine channels.\n"
        "3) Take the magnitude of FFT of the zero-padded prototype filter. This gives you the magnitude response of a coarse channel with fine channel resolution.\n"
        "4) Fold and sum the large magnitude response to represent the leakage from adjacent channels and aliasing from downsampling this channel. This is effectively a reshape of the large array to size [fine channels per coarse, coarse channels], then summing along the coarse channel dimension.\n"
        );
        man_page = man_page.append_section("EXAMPLES",
        "ATA: ./bliss_generate_channelizer_response -f 131072 -N 4 -M 2048 # some configurations use -f 262144\n"
        "GBT: ./bliss_generate_channelizer_response -f 1048576 -N 12 -M 256\n");

        std::cout << man_page;
        return 0;
    }

    // auto h_resp = bliss::gen_coarse_channel_response(131072, 2048, 4);
    auto h_resp = bliss::gen_coarse_channel_response(nchan_per_coarse, number_coarse_channels, taps_per_channel);

    bland::write_to_file(h_resp, out_file);
}
