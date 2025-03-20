
#include "file_types/dat_file.hpp"
#include "detail/raii_file_helpers.hpp"

#include <fmt/format.h>

#include <cstdio>
#include <ext/stdio_filebuf.h>
#include <ios>
#include <regex>
#include <string> // getline

using namespace bliss;


/**
 * Prettyprint (to a string) an angle in archours to sexagesimal
 * 
 * used especially for right ascenion (raj) from filterbank files
 */
std::string format_archours_to_sexagesimal(double src_raj) {
    double ra_deg = src_raj * 15.0;

    // Convert RA to sexagesimal
    int ra_hours = static_cast<int>(ra_deg / 15.0);
    int ra_minutes = static_cast<int>((ra_deg / 15.0 - ra_hours) * 60.0);
    double ra_seconds = ((ra_deg / 15.0 - ra_hours) * 60.0 - ra_minutes) * 60.0;

    return fmt::format("{:02}h{:02}m{:06.3f}s", ra_hours, ra_minutes, ra_seconds);
}

/**
 * Prettyprint (to a string) an angle in degrees to sexagesimal
 * 
 * used especially for declination (dej) from filterbank files
 */
std::string format_degrees_to_sexagesimal(double src_dej) {
    int dec_degrees = static_cast<int>(src_dej);
    int dec_arcminutes = static_cast<int>((std::abs(src_dej) - std::abs(dec_degrees)) * 60.0);
    double dec_arcseconds = ((std::abs(src_dej) - std::abs(dec_degrees)) * 60.0 - dec_arcminutes) * 60.0;
    char dec_sign = src_dej >= 0 ? '+' : '-';

    return fmt::format("{}{:02}d{:02}m{:05.2f}s", dec_sign, std::abs(dec_degrees), dec_arcminutes, dec_arcseconds);
}

template<typename Container>
std::string format_hits_to_dat_list(Container hits) {
    std::string table_string;
    auto hit_index = 1; // turbo-seti starts counting hits at 1, so might as well do the same
    for (auto this_hit : hits) {
        // # Top_Hit_#     Drift_Rate      SNR     Uncorrected_Frequency   Corrected_Frequency     Index   freq_start      freq_end        SEFD    SEFD_freq       Coarse_Channel_Number   Full_number_of_hits
        auto start_freq = this_hit.start_freq_MHz;
        auto end_freq = this_hit.start_freq_MHz + (this_hit.duration_sec * this_hit.drift_rate_Hz_per_sec)/1e6f;
        auto mid = (start_freq + end_freq)/2.0f;
        auto dat_line = fmt::format("{:06}\t{:4f}\t{:2f}\t{:6f}\t{:6f}\t{}\t{:6f}\t{:6f}\t{:1f}\t{:6f}\t{}\t{}\n",
                       hit_index++,
                       this_hit.drift_rate_Hz_per_sec,
                       this_hit.snr,
                       mid,
                       mid,
                       this_hit.start_freq_index, // This actually needs an adjustment to point to the mid bin...
                       this_hit.start_freq_MHz,
                       end_freq,
                       0.0,
                       0.0,
                       this_hit.coarse_channel_number,
                       this_hit.binwidth);
        table_string += dat_line;
    }
    return table_string;
}


// template<template<typename> class Container>
// void bliss::write_hits_to_dat_file(Container<hit> hits, std::string_view file_path) {
//     auto output_file = detail::raii_file_for_write(file_path);

//     auto table_contents = format_hits_to_dat_list(hits);
//     write(output_file._fd, table_contents.c_str(), table_contents.size());
// }

// std::list<hit> bliss::read_hits_from_dat_file(std::string_view file_path) {

// }

void bliss::write_scan_hits_to_dat_file(scan scan_with_hits, std::string_view file_path) {
    auto output_file = detail::raii_file_for_write(file_path);

    auto hits = scan_with_hits.hits();
    auto number_hits = hits.size();

    auto raj = scan_with_hits.src_raj();
    auto dej = scan_with_hits.src_dej();
    auto tstart = scan_with_hits.tstart();
    auto drift_range = scan_with_hits.get_drift_range();

    std::string file_path_id{"n/a"};
    try {
        file_path_id = scan_with_hits.get_file_path();
    } catch (std::exception &e) {
        fmt::print("WARN: file path not saved and is unknown\n");
    }

    std::string source_name{"n/a"};
    try {
        source_name = scan_with_hits.source_name();
    } catch (std::bad_optional_access &e) {
        fmt::print("WARN: source name is unknown\n");
    }

    auto formatted_raj = format_archours_to_sexagesimal(raj);
    auto formatted_dej = format_degrees_to_sexagesimal(dej);

    std::string header =
            fmt::format("# -------------------------- o --------------------------\n"
                        "# File ID: {}\n"
                        "# -------------------------- o --------------------------\n"
                        "# Source:{}\n"
                        "# MJD: {}\tRA: {}s\tDEC:{}\n"
                        "# DELTAT: {:6f}\tDELTAF(Hz):  {:6f} max_drift_rate: {}\tobs_length: {:2f}\n"
                        "# --------------------------\n"
                        "# "
                        "Top_Hit_#\tDrift_Rate\tSNR\tUncorrected_Frequency\tCorrected_Frequency\tIndex\tfreq_start\tfreq_end\tSEFD_freq\tCoarse_Channel_Number\tFull_number_of_hits\n"
                        "# --------------------------\n",
                        file_path_id,
                        source_name,
                        tstart,
                        formatted_raj,
                        formatted_dej,
                        scan_with_hits.tsamp(),
                        scan_with_hits.foff()*1e6,
                        drift_range.second,
                        scan_with_hits.ntsteps()*scan_with_hits.tsamp());
    write(output_file._fd, header.c_str(), header.size());
    auto table_contents = format_hits_to_dat_list(hits);
    write(output_file._fd, table_contents.c_str(), table_contents.size());
}

scan bliss::read_scan_hits_from_dat_file(std::string_view file_path) {
    auto in_file = detail::raii_file_for_read(file_path);

    auto in_buff   = __gnu_cxx::stdio_filebuf<char>(in_file._fd, std::ios::in);
    auto in_stream = std::istream(&in_buff);

    std::string line;
    std::regex  header_regex(R"(^#\s+MJD:\s+(\S+)\s+RA:\s+(\S+)\s+DEC:(\S+))");
    std::regex tstart_regex(R"(^#\s+DELTAT:\s+(\S+)\s+DELTAF\(Hz\):\s+(\S+)\s+max_drift_rate:\s+(\S+)\s+obs_length:\s+(\S+))");
    std::regex  data_regex(
            R"(^(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+))");

    scan deserialized_scan;
    while (std::getline(in_stream, line)) {
        std::smatch match;
        if (std::regex_search(line, match, header_regex)) {
            deserialized_scan.set_tstart(std::stod(match[1]));
            // scan_data.mjd = std::stod(match[1]);
            auto raj_sexagesimal = match[2];
            auto dej_sexagesimal = match[3];
        } else if (std::regex_search(line, match, tstart_regex)) {
            deserialized_scan.set_tsamp(std::stod(match[1]));
            deserialized_scan.set_foff(std::stod(match[2]));
            // scan_data.max_drift_rate = match[3];
            // scan_data.obs_length = std::stod(match[4]);
        } else if (std::regex_search(line, match, data_regex)) {
            hit new_hit;
            // hit index: match[1]
            new_hit.drift_rate_Hz_per_sec = std::stod(match[2]);
            new_hit.snr = std::stod(match[3]);
            // match[4]: "uncorrected frequency"
            // match[5]: "corrected frequency"
            new_hit.start_freq_index = std::stoi(match[6]);
            new_hit.start_freq_MHz = std::stod(match[7]);
            // match[8]: freq_end
            // match[9]: SEFD_freq
            new_hit.coarse_channel_number = std::stoi(match[10]);
            // match[11]: full_number_hits

            // TODO: need a way to push hits or list of hits back to a scan...
            // deserialized_scan.push_back(hit);
            // capnp does it like this:
            // scan(std::map<int, std::shared_ptr<coarse_channel>> coarse_channels);

        }
    }
    return deserialized_scan;
}
