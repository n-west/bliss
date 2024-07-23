
#include <preprocess/passband_static_equalize.hpp>

#include <bland/ndarray.hpp>
#include <bland/ndarray_slice.hpp>
#include <bland/ops/ops.hpp>

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

bland::ndarray bliss::firdes(int num_taps, float fc, std::string_view window) {
    // Hard code hamming for now
    auto h_prototype = sinc_lpf(fc, num_taps);
    auto w = hamming_window(num_taps); // TODO: look up other windows

    auto h = bland::ndarray({num_taps}, bland::ndarray::datatype::float32, bland::ndarray::dev::cpu);
    auto h_ptr = h.data_ptr<float>();
    for (int n=0; n < num_taps; ++n) {
        h_ptr[n] = h_prototype[n] *  w[n];
    }

    return h;
}

bland::ndarray bliss::gen_coarse_channel_response(int fine_per_coarse, int num_coarse_channels, int taps_per_channel, std::string window, std::string device_str) {
    // Get taps
    int num_taps = taps_per_channel * num_coarse_channels;

    // A typically channelizer for frequency analysis will specify cutoff to be
    // the channel width
    auto h = firdes(num_taps, 1.0f/static_cast<float>(num_coarse_channels), window);

    // Zero-pad to full rate (need to know the number of coarse channels in original recording)
    int total_fine_channels = num_coarse_channels * fine_per_coarse;

    int64_t full_res_length = static_cast<int64_t>(num_coarse_channels * fine_per_coarse);
    auto h_padded = bland::zeros({full_res_length});

    h_padded.slice(bland::slice_spec{0, 0, h.size(0)}) = h;

    // compute magnitude response (fft -> abs -> square)
    // h_padded = h_padded.to("cuda:0");
    // bland::write_to_file(h, "padded_h_from_firdes.cuda_f32");
    auto H = bland::fft_shift_mag_square(h_padded);

    // bland::write_to_file(H, "H_full_magsquare.cuda_f32");

    auto number_coarse_channels_contributing = H.size(0)/fine_per_coarse - 1;
    auto slice_dims = bland::slice_spec{0, fine_per_coarse/2, full_res_length-fine_per_coarse/2};
    auto H_slice = H.slice(slice_dims);
    H = H_slice;

    // bland::write_to_file(H_slice, "H_slice_magsquare.cuda_f32");
    // bland::write_to_file(H, "H_sliced_magsquare.cuda_f32");

    H.reshape({number_coarse_channels_contributing, fine_per_coarse});

    bland::write_to_file(H, "H_sliced_reshaped_magsquare.cuda_f32");

    H = bland::sum(H, {0}); // sum all of the energy from adjacent channels that folds back in
    H = H/bland::max(H); // normalize

    bland::write_to_file(H, "H_normalized.cuda_f32");

    return H;
}

coarse_channel bliss::equalize_passband_filter(coarse_channel cc, bland::ndarray h) {
    // data is (or can be) a deferred tensor
    // we might need to make a copy of cc rather than pass a ref since we're accessing the deferred array and setting it all in one go
    auto cc_ptr = std::make_shared<coarse_channel>(cc);
    cc.set_data(bland::ndarray_deferred([cc_data=cc_ptr, h](){
        return bland::divide(cc_data->data(), h);
    }));
    // cc.set_data(bland::divide(cc.data(), h));
    return cc;
}

coarse_channel bliss::equalize_passband_filter(coarse_channel cc, std::string_view h_resp_filepath, bland::ndarray::datatype dtype) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(cc, h);
}


scan bliss::equalize_passband_filter(scan sc, bland::ndarray h) {
    auto number_coarse_channels = sc.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = sc.read_coarse_channel(cc_index);
        *cc = equalize_passband_filter(*cc, h);
    }
    return sc;
}

scan bliss::equalize_passband_filter(scan sc, std::string_view h_resp_filepath, bland::ndarray::datatype dtype) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(sc, h);
}

observation_target bliss::equalize_passband_filter(observation_target ot, bland::ndarray h) {
    for (auto &scan_data : ot._scans) {
        scan_data = equalize_passband_filter(scan_data, h);
    }
    return ot;
}

observation_target bliss::equalize_passband_filter(observation_target ot, std::string_view h_resp_filepath, bland::ndarray::datatype dtype) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(ot, h);
}

cadence bliss::equalize_passband_filter(cadence ca, bland::ndarray h) {
    for (auto &target : ca._observations) {
        target = equalize_passband_filter(target, h);
    }
    return ca;
}

cadence bliss::equalize_passband_filter(cadence ca, std::string_view h_resp_filepath, bland::ndarray::datatype dtype) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(ca, h);
}
