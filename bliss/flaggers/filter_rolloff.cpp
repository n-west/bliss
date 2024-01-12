#include "flaggers/filter_rolloff.hpp"

#include <core/flag_values.hpp>

#include <bland/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

filterbank_data bliss::flag_filter_rolloff(filterbank_data fb_data, float rolloff_width) {
    auto &rfi_flags = fb_data.mask();

    int64_t one_sided_channels = std::round(fb_data.nchans() * rolloff_width);
    bland::slice(rfi_flags, {1, 0, one_sided_channels}) = bland::slice(rfi_flags, {1, 0, one_sided_channels}) + static_cast<uint8_t>(flag_values::filter_rolloff);
    bland::slice(rfi_flags, {1, -one_sided_channels, fb_data.nchans()}) = bland::slice(rfi_flags, {1, -one_sided_channels, fb_data.nchans()}) + static_cast<uint8_t>(flag_values::filter_rolloff);

    return fb_data;
}

observation_target bliss::flag_filter_rolloff(observation_target observations, float rolloff_width) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_filter_rolloff(filterbank, rolloff_width);
    }
    return observations;
}

cadence bliss::flag_filter_rolloff(cadence observations, float rolloff_width) {
    // TODO: it's probably unexpected that this would
    for (auto &observation : observations._observations) {
        observation = flag_filter_rolloff(observation, rolloff_width);
    }
    return observations;
}
