
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
                            uint8_t* rolloff_data,
                            uint8_t* low_sk_rfi_data, 
                            uint8_t* high_sk_rfi_data,
                            int32_t* drift_plane_shape, int32_t* drift_plane_strides,
                            float* spectrum_grid_data,
                            uint8_t* rfi_mask_data,
                            int32_t* spectrum_grid_shape, int32_t* spectrum_grid_strides,
                            kernel_drift_info* drifts, int32_t number_drifts, bool desmear) {

    extern __shared__ char smem[];
    kernel_drift_info* sdrifts = reinterpret_cast<kernel_drift_info*>(smem);
    auto smem_end = number_drifts * sizeof(kernel_drift_info);

    int32_t* sdrift_plane_strides = reinterpret_cast<int32_t*>(smem + smem_end);
    smem_end += sizeof(int32_t) * 2; // 2 dims

    int32_t* sspectrum_grid_strides = reinterpret_cast<int32_t*>(smem + smem_end);
    smem_end += sizeof(int32_t) * 2; // 2 dims

    for (int rate_index = threadIdx.x; rate_index < number_drifts + 8; rate_index += blockDim.x) {
        if (rate_index < number_drifts) {
            sdrifts[rate_index] = drifts[rate_index];
        } else if (rate_index < number_drifts + 2) {
            sdrift_plane_strides[rate_index-number_drifts] = drift_plane_strides[rate_index-number_drifts];
        } else if (rate_index < number_drifts + 4) {
            sspectrum_grid_strides[rate_index-number_drifts - 2] = spectrum_grid_strides[rate_index-number_drifts - 2];
        }
    }
    __syncthreads();


    // The strategy in this kernel is each thread does the entire dedrift for a single channel
    // and grid-strides until all channels are good

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    auto time_steps = spectrum_grid_shape[0];
    auto number_channels = spectrum_grid_shape[1];

    for (uint32_t freq_channel = tid; freq_channel < number_channels; freq_channel += grid_size) {
        for (uint32_t drift_index=0; drift_index < number_drifts; ++drift_index) {
            auto m = sdrifts[drift_index].slope;
            auto desmear_bandwidth = sdrifts[drift_index].desmeared_bins;
            
            int32_t drift_plane_index =
                    drift_index * sdrift_plane_strides[0] + freq_channel * sdrift_plane_strides[1];

            uint8_t accumulated_low_sk = 0;
            uint8_t accumulated_high_sk = 0;
            uint8_t accumulated_rolloff = 0;

            float accumulated_spectrum = 0;
            int accumulated_bins = 0;

            for (int t=0; t < time_steps; ++t) {
                int freq_offset_at_time = lroundf(m*t);

                for (int32_t desmear_channel=0; desmear_channel < desmear_bandwidth; ++desmear_channel) {
                    if (m >= 0) {
                        int32_t channel_offset = freq_offset_at_time + desmear_channel;

                        int32_t spectrum_freq_index = freq_channel + channel_offset;

                        if (spectrum_freq_index < number_channels && freq_channel < number_channels) {
                            int32_t spectrum_index  = t * sspectrum_grid_strides[0] + spectrum_freq_index * sspectrum_grid_strides[1];

                            accumulated_spectrum += spectrum_grid_data[spectrum_index];
                            accumulated_bins += 1;

                            auto rfi_val = rfi_mask_data[spectrum_index];
                            if (rfi_val & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) {
                                accumulated_low_sk += 1;
                            }
                            if (rfi_val & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) {
                                accumulated_high_sk += 1;
                            }
                            if (rfi_val & static_cast<uint8_t>(flag_values::filter_rolloff)) {
                                accumulated_rolloff += 1;
                            }
                        }

                    } else {
                        int channel_offset = freq_offset_at_time - desmear_channel;

                        int32_t spectrum_freq_index = freq_channel + channel_offset;

                        if (spectrum_freq_index > 0 && spectrum_freq_index < number_channels && freq_channel < number_channels) {
                            int32_t spectrum_index = t * sspectrum_grid_strides[0] + spectrum_freq_index * sspectrum_grid_strides[1];

                            accumulated_spectrum += spectrum_grid_data[spectrum_index];
                            accumulated_bins += 1;

                            auto rfi_val = rfi_mask_data[spectrum_index];
                            if (rfi_val & static_cast<uint8_t>(flag_values::low_spectral_kurtosis)) {
                                accumulated_low_sk += 1;
                            }
                            if (rfi_val & static_cast<uint8_t>(flag_values::high_spectral_kurtosis)) {
                                accumulated_high_sk += 1;
                            }
                            if (rfi_val & static_cast<uint8_t>(flag_values::filter_rolloff)) {
                                accumulated_rolloff += 1;
                            }
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
                rolloff_data[drift_plane_index] = accumulated_rolloff;
            }
        }
    }
}



[[nodiscard]] frequency_drift_plane
bliss::integrate_linear_rounded_bins_cuda(bland::ndarray    spectrum_grid,
                                         bland::ndarray    rfi_mask,
                                         integrate_drifts_options options) {
    auto spectrum_ptr     = spectrum_grid.data_ptr<float>();
    auto spectrum_shape   = spectrum_grid.shape();
    auto spectrum_strides = spectrum_grid.strides();

    auto rfi_ptr     = rfi_mask.data_ptr<uint8_t>();
    auto rfi_shape   = rfi_mask.shape();
    auto rfi_strides = rfi_mask.strides();

    auto number_drifts = (options.high_rate - options.low_rate) / options.rate_step_size;
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;
    std::vector<kernel_drift_info> device_rates;

    auto time_steps      = spectrum_grid.size(0);
    auto number_channels = spectrum_grid.size(1);

    auto maximum_drift_span = time_steps - 1;
    device_rates.reserve(number_drifts);
    drift_rate_info.reserve(number_drifts);
    for (int drift_index = 0; drift_index < number_drifts; ++drift_index) {
        // Drift in number of channels over the entire time extent
        auto drift_channels = options.low_rate + drift_index * options.rate_step_size;
        frequency_drift_plane::drift_rate rate;
        rate.index_in_plane = drift_index;

        // The actual slope of that drift (number channels / time)
        auto m = static_cast<float>(drift_channels) / static_cast<float>(maximum_drift_span);
        auto drift_info_for_device = kernel_drift_info();
        drift_info_for_device.slope = m;

        rate.drift_rate_slope = m;
        // If a single time step crosses more than 1 channel, there is smearing over multiple channels
        auto smeared_channels = std::round(std::abs(m));

        int desmear_bandwidth = 1;
        if (options.desmear) {
            desmear_bandwidth = std::max(1.0f, smeared_channels);
        }
        rate.desmeared_bins = desmear_bandwidth;
        drift_info_for_device.desmeared_bins = desmear_bandwidth;

        drift_rate_info.push_back(rate);
        device_rates.push_back(drift_info_for_device);
    }

    bland::ndarray drift_plane({number_drifts, number_channels}, spectrum_grid.dtype(), spectrum_grid.device());

    auto rfi_in_drift    = integrated_flags(number_drifts, number_channels, rfi_mask.device());
    auto rolloff_rfi_ptr = rfi_in_drift.filter_rolloff.data_ptr<uint8_t>();
    auto lowsk_rfi_ptr   = rfi_in_drift.low_spectral_kurtosis.data_ptr<uint8_t>();
    auto highsk_rfi_ptr  = rfi_in_drift.high_spectral_kurtosis.data_ptr<uint8_t>();

    auto rolloff_rfi_strides = rfi_in_drift.filter_rolloff.strides();
    auto lowsk_rfi_strides = rfi_in_drift.low_spectral_kurtosis.strides();
    auto highsk_rfi_strides = rfi_in_drift.high_spectral_kurtosis.strides();

    bland::fill(drift_plane, 0.0f);
    auto drift_plane_ptr     = drift_plane.data_ptr<float>();
    auto drift_plane_shape   = drift_plane.shape();
    auto drift_plane_strides = drift_plane.strides();


    thrust::device_vector<int32_t> dev_drift_plane_shape(drift_plane_shape.begin(), drift_plane_shape.end());
    thrust::device_vector<int32_t> dev_drift_plane_strides(drift_plane_strides.begin(), drift_plane_strides.end());

    thrust::device_vector<int32_t> dev_spectrum_shape(spectrum_shape.begin(), spectrum_shape.end());
    thrust::device_vector<int32_t> dev_spectrum_strides(spectrum_strides.begin(), spectrum_strides.end());

    thrust::device_vector<kernel_drift_info> dev_drift_slopes(device_rates.begin(), device_rates.end());

    int block_size = 512;
    int number_blocks = 112;
    auto smem = sizeof(kernel_drift_info) * number_drifts + sizeof(int32_t) * 4;
    integrate_drifts<<<number_blocks, block_size, smem>>>(
        drift_plane_ptr, rolloff_rfi_ptr,
        lowsk_rfi_ptr,
        highsk_rfi_ptr,
        thrust::raw_pointer_cast(dev_drift_plane_shape.data()), thrust::raw_pointer_cast(dev_drift_plane_strides.data()),
        spectrum_ptr,
        rfi_ptr,
        thrust::raw_pointer_cast(dev_spectrum_shape.data()), thrust::raw_pointer_cast(dev_spectrum_strides.data()),
        thrust::raw_pointer_cast(dev_drift_slopes.data()), number_drifts, options.desmear
    );

    // auto launch_ret = cudaDeviceSynchronize();
    // auto kernel_ret = cudaGetLastError();
    // if (launch_ret != cudaSuccess) {
    //     fmt::print("cuda launch got error {} ({})\n", launch_ret, cudaGetErrorString(launch_ret));
    // }
    // if (kernel_ret != cudaSuccess) {
    //     fmt::print("cuda launch got error {} ({})\n", kernel_ret, cudaGetErrorString(kernel_ret));
    // }

    // auto first_channel = 0; // This needs to be incrememnted by the offset from the most negative drift
    // We use all time available inside this function

    // normalize back by integration length
    frequency_drift_plane freq_drift(drift_plane, rfi_in_drift, time_steps, drift_rate_info);
    return freq_drift;
}

bland::ndarray bliss::integrate_linear_rounded_bins_cuda(bland::ndarray    spectrum_grid,
                                                        integrate_drifts_options options) {
    auto dummy_rfi_mask = bland::ndarray({1, 1});
    auto drift_plane = integrate_linear_rounded_bins_cuda(spectrum_grid, dummy_rfi_mask, options);
    return drift_plane.integrated_drift_plane();
}
