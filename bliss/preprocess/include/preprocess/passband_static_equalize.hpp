#pragma once

#include <bland/ndarray.hpp>
#include <bland/ndarray_deferred.hpp>
#include <core/cadence.hpp>
#include <core/coarse_channel.hpp>
#include <core/scan.hpp>

namespace bliss {

    bland::ndarray firdes(int num_taps, float fc);

    bland::ndarray gen_coarse_channel_inverse(int fine_per_coarse, int num_coarse_channels=2048, int taps_per_channel=4);

    coarse_channel equalize_passband_filter(coarse_channel cc, int num_coarse_channels, int taps_per_channel);

    // scan equalize_passband_filter(scan sc);

    // observation_target equalize_passband_filter(observation_target ot);

    // cadence equalize_passband_filter(cadence ca);

}