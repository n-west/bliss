
#include "file_types/dat_file.hpp"
#include "detail/raii_file_helpers.hpp"

#include <fmt/format.h>

#include <iostream>
#include <iomanip>

using namespace bliss;


template<template<typename> class Container>
void bliss::write_hits_to_dat_file(Container<hit> hits, std::string_view file_path) {

}

std::list<hit> bliss::read_hits_from_dat_file(std::string_view file_path) {

}

std::string format_archours_to_sexagesimal(double src_raj) {
    double ra_deg = src_raj * 15.0;

    // Convert RA to sexagesimal
    int ra_hours = static_cast<int>(ra_deg / 15.0);
    int ra_minutes = static_cast<int>((ra_deg / 15.0 - ra_hours) * 60.0);
    double ra_seconds = ((ra_deg / 15.0 - ra_hours) * 60.0 - ra_minutes) * 60.0;

    return fmt::format("{:02}h{:02}m{:06.3f}s", ra_hours, ra_minutes, ra_seconds);
}

std::string format_degrees_to_sexagesimal(double src_dej) {
    int dec_degrees = static_cast<int>(src_dej);
    int dec_arcminutes = static_cast<int>((std::abs(src_dej) - std::abs(dec_degrees)) * 60.0);
    double dec_arcseconds = ((std::abs(src_dej) - std::abs(dec_degrees)) * 60.0 - dec_arcminutes) * 60.0;
    char dec_sign = src_dej >= 0 ? '+' : '-';

    return fmt::format("{}{:02}d{:02}m{:05.2f}s", dec_sign, std::abs(dec_degrees), dec_arcminutes, dec_arcseconds);
}


void bliss::write_scan_hits_to_dat_file(scan scan_with_hits, std::string_view file_path) {
    auto output_file = detail::raii_file_for_write(file_path);

    auto hits = scan_with_hits.hits();
    auto number_hits = hits.size();
    auto hit_index = 0;
    output_file._fd;

    auto raj = scan_with_hits.src_raj();
    auto dej = scan_with_hits.src_dej();
    auto tstart = scan_with_hits.tstart();

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
                        scan_with_hits.get_file_path(),
                        scan_with_hits.source_name(),
                        scan_with_hits.tstart(),
                        formatted_raj,
                        formatted_dej,
                        scan_with_hits.tsamp(),
                        scan_with_hits.foff()*1e6,
                        "n/a",
                        scan_with_hits.ntsteps()*scan_with_hits.tsamp());
    write(output_file._fd, header.c_str(), header.size());

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
        write(output_file._fd, dat_line.c_str(), dat_line.size());
    }
}

scan bliss::read_scan_hits_from_dat_file(std::string_view file_path) {

}
