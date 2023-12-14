#include "flaggers/filter_rolloff.hpp"

#include <bland/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

filterbank_data bliss::flag_filter_rolloff(filterbank_data fb_data, float rolloff_width) {
    auto &rfi_flags = fb_data.mask();

    int64_t one_sided_channels = std::round(fb_data.nchans() * rolloff_width);
    bland::fill(bland::slice(rfi_flags, {1, 0, one_sided_channels}), 2);
    bland::fill(bland::slice(rfi_flags, {1, -one_sided_channels, fb_data.nchans()}), 2);

    return fb_data;
}