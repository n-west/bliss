#include "flaggers/filter_rolloff.hpp"

#include <core/flag_values.hpp>

#include <bland/ops/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

coarse_channel bliss::flag_filter_rolloff(coarse_channel cc_data, float rolloff_width) {
    auto rfi_flags = cc_data.mask();

    int64_t one_sided_channels = std::round(cc_data.nchans() * rolloff_width);
    bland::slice(rfi_flags, {1, 0, one_sided_channels}) = bland::slice(rfi_flags, {1, 0, one_sided_channels}) + static_cast<uint8_t>(flag_values::filter_rolloff);
    bland::slice(rfi_flags, {1, -one_sided_channels, cc_data.nchans()}) = bland::slice(rfi_flags, {1, -one_sided_channels, cc_data.nchans()}) + static_cast<uint8_t>(flag_values::filter_rolloff);
    cc_data.set_mask(rfi_flags);
    return cc_data;
}


scan bliss::flag_filter_rolloff(scan fil_data, float rolloff_width) {
    auto number_coarse_channels = fil_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = fil_data.get_coarse_channel(cc_index);
        *cc = flag_filter_rolloff(*cc, rolloff_width);
    }
    return fil_data;
}

observation_target bliss::flag_filter_rolloff(observation_target observations, float rolloff_width) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_filter_rolloff(filterbank, rolloff_width);
    }
    return observations;
}

cadence bliss::flag_filter_rolloff(cadence observations, float rolloff_width) {
    for (auto &observation : observations._observations) {
        observation = flag_filter_rolloff(observation, rolloff_width);
    }
    return observations;
}
