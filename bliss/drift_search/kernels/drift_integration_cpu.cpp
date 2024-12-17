
#include "drift_integration_cpu.hpp"

#include "core/frequency_drift_plane.hpp"

#include "bland/ops/ops.hpp" // fill

#include <fmt/core.h>
#include <fmt/format.h>

#include <cmath> // std::round, std::abs

using namespace bliss;

[[nodiscard]] frequency_drift_plane
bliss::integrate_linear_rounded_bins_cpu(bland::ndarray    spectrum_grid,
                                         bland::ndarray    rfi_mask,
                                         std::vector<frequency_drift_plane::drift_rate> drift_rates,
                                         integrate_drifts_options options) {
    auto spectrum_ptr     = spectrum_grid.data_ptr<float>();
    auto spectrum_strides = spectrum_grid.strides();
    auto spectrum_shape   = spectrum_grid.shape();

    auto rfi_ptr     = rfi_mask.data_ptr<uint8_t>();
    auto rfi_strides = rfi_mask.strides();
    auto rfi_shape   = rfi_mask.shape();

    int32_t time_steps      = spectrum_grid.size(0);
    auto number_channels = spectrum_grid.size(1);
    auto number_drifts = static_cast<int64_t>(drift_rates.size());

    bland::ndarray drift_plane({number_drifts, number_channels}, spectrum_grid.dtype(), spectrum_grid.device());

    auto rfi_in_drift    = integrated_flags(number_drifts, number_channels, rfi_mask.device());
    auto sigma_clip_rfi_ptr = rfi_in_drift.sigma_clip.data_ptr<uint8_t>();
    auto lowsk_rfi_ptr   = rfi_in_drift.low_spectral_kurtosis.data_ptr<uint8_t>();
    auto highsk_rfi_ptr  = rfi_in_drift.high_spectral_kurtosis.data_ptr<uint8_t>();

    auto sigma_clip_rfi_strides = rfi_in_drift.sigma_clip.strides();
    auto lowsk_rfi_strides = rfi_in_drift.low_spectral_kurtosis.strides();
    auto highsk_rfi_strides = rfi_in_drift.high_spectral_kurtosis.strides();

    bland::fill(drift_plane, 0.0f);
    auto drift_plane_ptr     = drift_plane.data_ptr<float>();
    auto drift_plane_strides = drift_plane.strides();
    auto drift_plane_shape   = drift_plane.shape();

    auto first_channel = 0; // This needs to be incrememnted by the offset from the most negative drift
    // We use all time available inside this function
    for (int drift_index = 0; drift_index < number_drifts; ++drift_index) {
        auto rate = drift_rates[drift_index];
        // The actual slope of that drift (number channels / time)
        auto m = rate.drift_rate_slope;
        auto desmear_bins = rate.desmeared_bins;

        for (int t = 0; t < time_steps; ++t) {
            int freq_offset_at_time  = std::round(m * t);
            // int freq_offset_at_time2 = std::round(m * (t + time_steps / 2));

            for (int desmear_channel = 0; desmear_channel < desmear_bins; ++desmear_channel) {
                if (m >= 0) {
                    // The accumulator (drift spectrum) stays fixed at 0 while the spectrum start increments
                    auto channel_offset  = freq_offset_at_time + desmear_channel;

                    int64_t drift_freq_slice_start = 0;
                    int64_t drift_freq_slice_end   = number_channels - channel_offset;

                    int64_t spectrum_freq_slice_start  = channel_offset;
                    int64_t spectrum_freq_slice_end    = number_channels;
                    if (spectrum_freq_slice_start > spectrum_shape[1]) {
                        fmt::print("ERROR: drift integration might be going out of bounds. Report this condition");
                    }

                    auto   number_channels = drift_freq_slice_end - drift_freq_slice_start;
                    size_t drift_plane_index =
                            drift_index * drift_plane_strides[0] + drift_freq_slice_start * drift_plane_strides[1];
                    size_t spectrum_index  = t * spectrum_strides[0] + spectrum_freq_slice_start * spectrum_strides[1];

                    size_t lowsk_index = drift_index * lowsk_rfi_strides[0] + drift_freq_slice_start * lowsk_rfi_strides[1];
                    size_t highsk_index = drift_index * highsk_rfi_strides[0] + drift_freq_slice_start * highsk_rfi_strides[1];
                    size_t filtrolloff_index = drift_index * sigma_clip_rfi_strides[0] + drift_freq_slice_start * sigma_clip_rfi_strides[1];
                    size_t rfi_index = t * rfi_strides[0] + spectrum_freq_slice_start * rfi_strides[1];
                    for (size_t channel = 0; channel < number_channels; ++channel) {
                        drift_plane_ptr[drift_plane_index] += spectrum_ptr[spectrum_index] / desmear_bins;
                        drift_plane_index += drift_plane_strides[1];
                        spectrum_index += spectrum_strides[1];
                        if (collect_rfi) {
                            if (rfi_ptr[rfi_index] & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) {
                                lowsk_rfi_ptr[lowsk_index] += 1;
                            }
                            lowsk_index += lowsk_rfi_strides[1];
                            if (rfi_ptr[rfi_index] & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) {
                                highsk_rfi_ptr[highsk_index] += 1;
                            }
                            highsk_index += highsk_rfi_strides[1];
                            if (rfi_ptr[rfi_index] & static_cast<uint8_t>(flag_values::filter_rolloff)) {
                                sigma_clip_rfi_ptr[filtrolloff_index] += 1;
                            }
                            filtrolloff_index += sigma_clip_rfi_strides[1];

                            rfi_index += rfi_strides[1];
                        }
                    }
                } else {
                    // At a negative drift rate, everything needs to scooch up instead of chop down
                    // the desmeared channel might need to be negative as well (it's the channel we're advancing
                    // towards)
                    auto channel_offset = freq_offset_at_time - desmear_channel;

                    int64_t drift_freq_slice_start = -rate.drift_channels_span;
                    int64_t drift_freq_slice_end   = number_channels;

                    int64_t spectrum_freq_slice_start = -rate.drift_channels_span + channel_offset;
                    int64_t spectrum_freq_slice_end   = number_channels + channel_offset;
                    if (spectrum_freq_slice_start < 0) {
                        // This condition occurs at fast drive rates (very negative) with desmearing on.
                        auto offset_amount = spectrum_freq_slice_start;

                        spectrum_freq_slice_start -= offset_amount;
                        drift_freq_slice_start -= offset_amount;
                    }

                    auto   number_channels = drift_freq_slice_end - drift_freq_slice_start;
                    size_t drift_plane_index =
                            drift_index * drift_plane_strides[0] + drift_freq_slice_start * drift_plane_strides[1];
                    size_t spectrum_index = t * spectrum_strides[0] + spectrum_freq_slice_start * spectrum_strides[1];

                    size_t lowsk_index = drift_index * lowsk_rfi_strides[0] + drift_freq_slice_start * lowsk_rfi_strides[1];
                    size_t highsk_index = drift_index * highsk_rfi_strides[0] + drift_freq_slice_start * highsk_rfi_strides[1];
                    size_t sigma_clip_index = drift_index * sigma_clip_rfi_strides[0] + drift_freq_slice_start * sigma_clip_rfi_strides[1];
                    size_t rfi_index = t * rfi_strides[0] + spectrum_freq_slice_start * rfi_strides[1];
                    for (size_t channel = 0; channel < number_channels; ++channel) {
                        drift_plane_ptr[drift_plane_index] += spectrum_ptr[spectrum_index] / desmear_bins;
                        drift_plane_index += drift_plane_strides[1];
                        spectrum_index += spectrum_strides[1];
                        if (collect_rfi) {
                            if (rfi_ptr[rfi_index] & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) {
                                lowsk_rfi_ptr[lowsk_index] += 1;
                            }
                            lowsk_index += lowsk_rfi_strides[1];
                            if (rfi_ptr[rfi_index] & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) {
                                highsk_rfi_ptr[highsk_index] += 1;
                            }
                            highsk_index += highsk_rfi_strides[1];
                            if (rfi_ptr[rfi_index] & static_cast<uint8_t>(flag_values::filter_rolloff)) {
                                sigma_clip_rfi_ptr[sigma_clip_index] += 1;
                            }
                            sigma_clip_index += sigma_clip_rfi_strides[1];

                            rfi_index += rfi_strides[1];
                        }
                    }
                }
            }
        }
    }

    // normalize back by integration length
    frequency_drift_plane freq_drift(drift_plane / time_steps, rfi_in_drift, time_steps, drift_rates);
    return freq_drift;
}

// bland::ndarray bliss::integrate_linear_rounded_bins_cpu(bland::ndarray    spectrum_grid,
//                                                         integrate_drifts_options options) {
//     auto dummy_rfi_mask = bland::ndarray({1, 1});
//     auto drift_plane = integrate_linear_rounded_bins_cpu(spectrum_grid, dummy_rfi_mask, options);
//     return drift_plane.integrated_drift_plane();
// }
