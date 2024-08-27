
#include <file_types/cpnp_files.hpp>
#include <file_types/dat_file.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>
#include <cstdint>
#include <string>
#include <vector>

#include <chrono>
#include <iostream>

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

    std::string input_file;
    std::string output_path = "";

    bool help = false;
    auto cli = (
        (
            (clipp::option("-i", "--input") & clipp::value("input_file").set(input_file)) % "input files with scan hits, expected to be capnp serialized",
            (clipp::option("-o", "--output") & (clipp::value("output_file").set(output_path))) % "path to output file"
        )
        |
        clipp::option("-h", "--help").set(help) % "Show this screen."
    );

    auto parse_result = clipp::parse(argc, argv, cli);

    if (input_file.empty()) {
        fmt::print("Missing an input file. Use '-i' or '--input' to provide a file to convert.\n");
        help = true;
    }
    if (!parse_result || help) {
        std::cout << clipp::make_man_page(cli, "bliss_hits_to_dat");
        return 0;
    }

    if (output_path.empty()) {
        auto path = fs::path(input_file);
        output_path = path.filename().replace_extension("dat");
    }

    // How can we discover which type of capnp message/file this is?
    auto scan_with_hits = bliss::read_scan_hits_from_capnp_file(input_file);
    auto hits = scan_with_hits.hits();
    if (output_path == "-" || output_path == "stdout") {
        fmt::print("Got {} hits\n", hits.size());
        for (auto &h : hits) {
            std::cout << h.repr() << std::endl;
        }
    } else {
        bliss::write_scan_hits_to_dat_file(scan_with_hits, output_path);
    }

}
