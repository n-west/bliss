#pragma once

#include <bland/ndarray.hpp>
#include <core/cadence.hpp>
#include <core/coarse_channel.hpp>
#include <core/scan.hpp>

namespace bliss {

    bland::ndarray firdes(int num_taps, float fc, std::string_view window);

    bland::ndarray gen_coarse_channel_response(int fine_per_coarse, int num_coarse_channels=2048, int taps_per_channel=4, std::string window="hamming", std::string device_str="cpu");

    coarse_channel equalize_passband_filter(coarse_channel cc, bland::ndarray h, bool validate=false);

    coarse_channel equalize_passband_filter(coarse_channel cc, std::string_view h_resp_filepath, bland::ndarray::datatype dtype=bland::ndarray::datatype::float32, bool validate=false);

    scan equalize_passband_filter(scan sc, bland::ndarray h, bool validate=false);

    scan equalize_passband_filter(scan sc, std::string_view h_resp_filepath, bland::ndarray::datatype dtype=bland::ndarray::datatype::float32, bool validate=false);

    observation_target equalize_passband_filter(observation_target ot, bland::ndarray h, bool validate=false);

    observation_target equalize_passband_filter(observation_target ot, std::string_view h_resp_filepath, bland::ndarray::datatype dtype=bland::ndarray::datatype::float32, bool validate=false);

    cadence equalize_passband_filter(cadence ca, bland::ndarray h, bool validate=false);

    cadence equalize_passband_filter(cadence ca, std::string_view h_resp_filepath, bland::ndarray::datatype dtype=bland::ndarray::datatype::float32, bool validate=false);

}