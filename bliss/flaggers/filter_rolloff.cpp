#include "flaggers/filter_rolloff.hpp"

#include <core/flag_values.hpp>

#include <bland/ops/ops.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

using namespace bliss;

coarse_channel bliss::flag_filter_rolloff(coarse_channel cc_data, float rolloff_width) {

    auto deferred_rfi_mask = bland::ndarray_deferred([mask=cc_data.mask(), rolloff_width]() mutable {
        bland::ndarray rfi_flags = mask;

        auto nchans = rfi_flags.shape()[1];
        int64_t one_sided_channels = std::round(nchans * rolloff_width);
        bland::slice(rfi_flags, {1, 0, one_sided_channels}) = bland::slice(rfi_flags, {1, 0, one_sided_channels}) + static_cast<uint8_t>(flag_values::filter_rolloff);
        bland::slice(rfi_flags, {1, -one_sided_channels, nchans}) = bland::slice(rfi_flags, {1, -one_sided_channels, nchans}) + static_cast<uint8_t>(flag_values::filter_rolloff);
        return rfi_flags;
    });
    
    cc_data.set_mask(deferred_rfi_mask);
    return cc_data;
}


scan bliss::flag_filter_rolloff(scan fil_data, float rolloff_width) {
    fil_data.add_coarse_channel_transform([rolloff_width](coarse_channel cc) {
        return flag_filter_rolloff(cc, rolloff_width);
    });
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
