
#include <preprocess/passband_static_equalize.hpp>

#include <bland/ndarray.hpp>
#include <bland/ndarray_slice.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>

using namespace bliss;

std::vector<float> hamming_window(int N) {
    auto N_float = static_cast<float>(N-1);
    constexpr float two_pi = 2 * M_PI;
    constexpr float a_0 = 0.54;
    constexpr float a_1 = 0.46;
    auto w = std::vector<float>();
    w.reserve(N);
    for (int n = 0; n < N; ++n) {
        w.push_back(a_0 - a_1 * cos((two_pi * n) / N_float));
    }

    return w;
}

float sinc(float t) {
    auto x = M_PI*t;
    if (std::fabs(x) < 0.01f) {
        return cos(x/2.0f) * cos(x/4.0f) * cos(x/8.0f);
    } else {
        return sin(x) / (x);
    }
}

std::vector<float> sinc_lpf(float fc, int num_taps) {
    // fc in range 0..1

    std::vector<float> prototype_lpf;
    prototype_lpf.reserve(num_taps);
    auto half_taps = static_cast<float>(num_taps-1)/2.0f;
    for (int n=0; n < num_taps; ++n) {
        auto t = static_cast<float>(n) - half_taps;
        prototype_lpf.emplace_back(sinc(fc*t));
    }
    return prototype_lpf;
}

bland::ndarray bliss::firdes(int num_taps, float fc) {
    // Hard code hamming for now
    auto h_prototype = sinc_lpf(fc, num_taps);
    auto w = hamming_window(num_taps);

    auto h = bland::ndarray({num_taps}, bland::ndarray::datatype::float32, bland::ndarray::dev::cpu);
    auto h_ptr = h.data_ptr<float>();
    for (int n=0; n < num_taps; ++n) {
        h_ptr[n] = h_prototype[n] *  w[n];
    }

    return h;
}

bland::ndarray bliss::gen_coarse_channel_inverse(int fine_per_coarse, int num_coarse_channels, int taps_per_channel) {
    // Get taps
    // int num_coarse_channels = 2048;
    // int taps_per_channel = 4;
    int num_taps = taps_per_channel * num_coarse_channels;

    // A typically channelizer for frequency analysis will specify cutoff to be
    // the channel width
    auto h = firdes(num_taps, 1.0f/static_cast<float>(num_coarse_channels));

    // Zero-pad to full rate (need to know the number of coarse channels in original recording)
    int total_fine_channels = num_coarse_channels * fine_per_coarse;

    int64_t full_res_length = num_coarse_channels * fine_per_coarse;
    auto h_padded = bland::zeros({full_res_length});
    fmt::print("Slicing {} - {}\n", 0, h.size(0));
    h_padded.slice(bland::slice_spec{0, 0, h.size(0)}) = h;
    fmt::print("h_padded repr: {}\n", h_padded.repr());
    // compute magnitude response (fft -> abs -> square)
    // h_padded = h_padded.to("cpu");
    h_padded = h_padded.to("cuda:0");
    auto H = bland::fft_shift_mag_square(h_padded);
    fmt::print("Done with the fft_shift_mag_square... doing slice and reshape\n");
    H = H.to("cpu");

    auto number_coarse_channels_contributing = H.size(0)/fine_per_coarse - 1;
    H = H.slice(bland::slice_spec{0, fine_per_coarse/2, full_res_length-fine_per_coarse/2});
    fmt::print("H_slice is like {}\n", H.shape());
    H = H.to("cuda:0");
    H = H.reshape({number_coarse_channels_contributing, fine_per_coarse});
    fmt::print("H_slice reshaped is like {}\n", H.shape());
    // H = H.slice(bland::slice_spec{0, fine_per_coarse/2, -fine_per_coarse/2+1}).reshape({number_coarse_channels_contributing, fine_per_coarse});
    H = bland::sum(H, {0});

    return H;
}

coarse_channel bliss::equalize_passband_filter(coarse_channel cc, int num_coarse_channels, int taps_per_channel) {
    // Get taps
    // int num_coarse_channels = 2048;
    // int taps_per_channel = 4;
    int num_taps = taps_per_channel * num_coarse_channels;

    // A typically channelizer for frequency analysis will specify cutoff to be
    // the channel width
    auto h = firdes(num_taps, 1.0f/static_cast<float>(num_coarse_channels));

    // Zero-pad to full rate (need to know the number of coarse channels in original recording)
    auto fine_per_coarse = cc.nchans();
    // This is the resolution of this filter at the full-rate. We want to see the response at each fine channel
    int total_fine_channels = num_coarse_channels * fine_per_coarse;

    auto h_padded = bland::zeros({num_coarse_channels * fine_per_coarse});
    h_padded.slice(bland::slice_spec{0, 0, h.size(0)}) = h;
    // compute magnitude response (fft -> abs -> square)
    h_padded = h_padded.to("cuda:0");
    auto H = bland::fft_shift_mag_square(h_padded);


    // reshape and sum along channels
    /// apply that inverse

}