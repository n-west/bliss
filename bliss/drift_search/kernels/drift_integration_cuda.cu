
#include "drift_integration_cuda.cuh"

#include "core/frequency_drift_plane.hpp"

#include "bland/ops/ops.hpp" // fill

#include <fmt/core.h>
#include <fmt/format.h>

#include <thrust/device_vector.h>
// #include <cuda_runtime.h>


#include <cmath> // std::round, std::abs

using namespace bliss;

/**
 * This kernel will parallelize such that a single cuda thread handles integration for all drifts
 * of a frequency channel.
*/

struct kernel_drift_info {
    double slope;
    int desmeared_bins;
};

__global__ void integrate_drifts(float* drift_plane_data,
                            uint8_t* sigmaclip_data,
                            uint8_t* low_sk_rfi_data, 
                            uint8_t* high_sk_rfi_data,
                            int32_t* drift_plane_shape, int32_t* drift_plane_strides,
                            const float* spectrum_grid_data,
                            const uint8_t* rfi_mask_data,
                            int32_t* spectrum_grid_shape, int32_t* spectrum_grid_strides,
                            frequency_drift_plane::drift_rate* drifts, int32_t number_drifts, bool desmear) {
    // The strategy in this kernel is each thread does the entire dedrift for a single channel
    // and grid-strides until all channels are good

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    auto time_steps = spectrum_grid_shape[0];
    auto number_channels = spectrum_grid_shape[1];

    for (uint64_t freq_channel = tid; freq_channel < number_channels; freq_channel += grid_size) {
        for (uint64_t drift_index=0; drift_index < number_drifts; ++drift_index) {
            auto m = drifts[drift_index].drift_rate_slope;
            auto desmear_bandwidth = drifts[drift_index].desmeared_bins;
            
            int64_t drift_plane_index = drift_index * number_channels + freq_channel;

            uint8_t accumulated_low_sk = 0;
            uint8_t accumulated_high_sk = 0;
            uint8_t accumulated_sigmaclip = 0;

            float accumulated_spectrum = 0;
            int accumulated_bins = 0;

            for (int t=0; t < time_steps; ++t) {
                int freq_offset_at_time = lroundf(m * t); // many round modes available, we want to round-away from zero

                for (int32_t desmear_channel = 0; desmear_channel < desmear_bandwidth; ++desmear_channel) {
                    int32_t channel_offset = (m >= 0) ? (freq_offset_at_time + desmear_channel) : (freq_offset_at_time - desmear_channel);
                    int32_t spectrum_freq_index = freq_channel + channel_offset;

                    if (spectrum_freq_index >= 0 && spectrum_freq_index < number_channels) {
                        int64_t spectrum_index = t * number_channels + spectrum_freq_index;

                        accumulated_spectrum += spectrum_grid_data[spectrum_index];
                        accumulated_bins += 1;

                        auto rfi_val = rfi_mask_data[spectrum_index];
                        if (rfi_val & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) {
                            accumulated_low_sk += 1;
                        }
                        if (rfi_val & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) {
                            accumulated_high_sk += 1;
                        }
                        if (rfi_val & static_cast<uint8_t>(flag_values::sigma_clip)) {
                            accumulated_sigmaclip += 1;
                        }
                    }
                }
            }

            if (accumulated_bins == 0) {
                drift_plane_data[drift_plane_index] = 0;
            } else {
                drift_plane_data[drift_plane_index] = accumulated_spectrum / accumulated_bins;
                low_sk_rfi_data[drift_plane_index] = accumulated_low_sk;
                high_sk_rfi_data[drift_plane_index] = accumulated_high_sk;
                sigmaclip_data[drift_plane_index] = accumulated_sigmaclip;
            }
        }
    }
}


template <typename T, typename It>
thrust::device_vector<T> safe_device_vector(It begin_alloc, It end_alloc) {
    try {
        return thrust::device_vector<T>(begin_alloc, end_alloc);
    }
    catch (std::exception &e) {
        fmt::print("Error: {}\n", e.what());
        throw std::runtime_error("Memory allocation failed for thrust::device_vector");
    }
}

#include <thread>

[[nodiscard]] frequency_drift_plane
bliss::integrate_linear_rounded_bins_cuda(bland::ndarray    spectrum_grid,
                                         bland::ndarray    rfi_mask,
                                         std::vector<frequency_drift_plane::drift_rate> drift_rates,
                                         integrate_drifts_options options) {
    auto spectrum_ptr     = spectrum_grid.data_ptr<float>();
    auto spectrum_shape   = spectrum_grid.shape();
    auto spectrum_strides = spectrum_grid.strides();

    auto rfi_ptr     = rfi_mask.data_ptr<uint8_t>();
    auto rfi_shape   = rfi_mask.shape();
    auto rfi_strides = rfi_mask.strides();

    std::vector<kernel_drift_info> device_rates;

    auto time_steps      = spectrum_grid.size(0);
    auto number_channels = spectrum_grid.size(1);

    auto number_drifts = static_cast<int64_t>(drift_rates.size());

    bland::ndarray drift_plane({number_drifts, number_channels}, spectrum_grid.dtype(), spectrum_grid.device());

    auto rfi_in_drift    = integrated_flags(number_drifts, number_channels, rfi_mask.device());
    auto sigmaclip_rfi_ptr = rfi_in_drift.sigma_clip.data_ptr<uint8_t>();
    auto lowsk_rfi_ptr   = rfi_in_drift.low_spectral_kurtosis.data_ptr<uint8_t>();
    auto highsk_rfi_ptr  = rfi_in_drift.high_spectral_kurtosis.data_ptr<uint8_t>();

    auto sigmaclip_rfi_strides = rfi_in_drift.sigma_clip.strides();
    auto lowsk_rfi_strides = rfi_in_drift.low_spectral_kurtosis.strides();
    auto highsk_rfi_strides = rfi_in_drift.high_spectral_kurtosis.strides();

    bland::fill(drift_plane, 0.0f);
    auto drift_plane_ptr     = drift_plane.data_ptr<float>();
    auto drift_plane_shape   = drift_plane.shape();
    auto drift_plane_strides = drift_plane.strides();

    auto dev_drift_plane_shape = safe_device_vector<int32_t>(drift_plane_shape.begin(), drift_plane_shape.end());
    auto dev_drift_plane_strides= safe_device_vector<int32_t>(drift_plane_strides.begin(), drift_plane_strides.end());

    auto dev_spectrum_shape = safe_device_vector<int32_t>(spectrum_shape.begin(), spectrum_shape.end());
    auto dev_spectrum_strides = safe_device_vector<int32_t>(spectrum_strides.begin(), spectrum_strides.end());

    auto dev_drift_slopes = safe_device_vector<frequency_drift_plane::drift_rate>(drift_rates.begin(), drift_rates.end());

    dim3 grid(4096, 1);
    dim3 block(256, 1);
    integrate_drifts<<<grid, block>>>(
        drift_plane_ptr,
        sigmaclip_rfi_ptr,
        lowsk_rfi_ptr,
        highsk_rfi_ptr,
        thrust::raw_pointer_cast(dev_drift_plane_shape.data()), thrust::raw_pointer_cast(dev_drift_plane_strides.data()),
        spectrum_ptr,
        rfi_ptr,
        thrust::raw_pointer_cast(dev_spectrum_shape.data()), thrust::raw_pointer_cast(dev_spectrum_strides.data()),
        thrust::raw_pointer_cast(dev_drift_slopes.data()), number_drifts, options.desmear
    );

    auto launch_ret = cudaDeviceSynchronize();
    auto kernel_ret = cudaGetLastError();
    if (launch_ret != cudaSuccess) {
        fmt::print("cuda launch got error {} ({})\n", static_cast<int>(launch_ret), cudaGetErrorString(launch_ret));
    }
    if (kernel_ret != cudaSuccess) {
        fmt::print("cuda launch got error {} ({})\n", static_cast<int>(kernel_ret), cudaGetErrorString(kernel_ret));
    }

    // normalize back by integration length
    frequency_drift_plane freq_drift(drift_plane, rfi_in_drift, drift_rates);
    return freq_drift;
}

