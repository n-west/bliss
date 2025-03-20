
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

    // bland::write_to_file(H, "H_sliced_reshaped_magsquare.cuda_f32");

    H = bland::sum(H, {0}); // sum all of the energy from adjacent channels that folds back in
    H = H/bland::max(H); // normalize

    // bland::write_to_file(H, "H_normalized.cuda_f32");

    return H;
}

coarse_channel bliss::equalize_passband_filter(coarse_channel cc, bland::ndarray H, bool validate) {
    // Validate that the mean of the passband is 2x the mean of the cutoff
    // TODO: this could check the ratios of the filter response and force them to be equal

    if (validate) {
        auto spectra = cc.data();
        auto num_channels = spectra.size(1);
        bland::ndarray spectra_passband = bland::mean(spectra.slice(bland::slice_spec{1, num_channels/2 - 100, num_channels / 2 + 100}));
        bland::ndarray spectra_cutoff_lower = bland::mean(spectra.slice(bland::slice_spec{1, 0, 50}));
        bland::ndarray spectra_cutoff_upper = bland::mean(spectra.slice(bland::slice_spec{1, -50, -1}));
        auto spectra_passband_mean = spectra_passband.scalarize<float>();
        auto spectra_cutoff_upper_mean = spectra_cutoff_lower.scalarize<float>();
        auto spectra_cutoff_lower_mean = spectra_cutoff_upper.scalarize<float>();
        auto spectra_cutoff_mean = (spectra_cutoff_upper_mean + spectra_cutoff_lower_mean) / 2.0f;

        // H should be one dimensional with the same number of channels as given spectra
        bland::ndarray H_passband = bland::mean(H.slice(bland::slice_spec{0, num_channels/2 - 100, num_channels / 2 + 100}));
        bland::ndarray H_cutoff_lower = bland::mean(H.slice(bland::slice_spec{0, 0, 50}));
        bland::ndarray H_cutoff_upper = bland::mean(H.slice(bland::slice_spec{0, -50, -1}));
        auto H_passband_mean = H_passband.scalarize<float>();
        auto H_cutoff_upper_mean = H_cutoff_lower.scalarize<float>();
        auto H_cutoff_lower_mean = H_cutoff_upper.scalarize<float>();
        auto H_ratio = H_passband_mean / ((H_cutoff_upper_mean + H_cutoff_lower_mean) / 2.0f);

        if (spectra_cutoff_mean > 1.05 * (spectra_passband_mean / H_ratio)) {
            fmt::print("WARN: the ratio ({}) of power between the passband ({}) and cutoff ({}) for data does not "
                       "match given ratio of power in the given filter response ({}). This is potentially a data "
                       "quality issue inherent to recording that will cause flared edges after correction which can "
                       "lead to false positives.\n",
                       spectra_passband_mean / spectra_cutoff_mean,
                       spectra_passband_mean,
                       spectra_cutoff_mean,
                       H_ratio);
        }
    }
    H = H.to(cc.device());
    cc.set_data(bland::divide(cc.data(), H));
    return cc;
}

coarse_channel bliss::equalize_passband_filter(coarse_channel cc, std::string_view h_resp_filepath, bland::ndarray::datatype dtype, bool validate) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(cc, h, validate);
}


scan bliss::equalize_passband_filter(scan sc, bland::ndarray h, bool validate) {
    auto number_coarse_channels = sc.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        sc.add_coarse_channel_transform([h, validate](coarse_channel cc) { return equalize_passband_filter(cc, h, validate); });
        // auto cc = sc.read_coarse_channel(cc_index);
        // *cc = equalize_passband_filter(*cc, h);
    }
    return sc;
}

scan bliss::equalize_passband_filter(scan sc, std::string_view h_resp_filepath, bland::ndarray::datatype dtype, bool validate) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(sc, h, validate);
}

observation_target bliss::equalize_passband_filter(observation_target ot, bland::ndarray h, bool validate) {
    for (auto &scan_data : ot._scans) {
        scan_data = equalize_passband_filter(scan_data, h, validate);
    }
    return ot;
}

observation_target bliss::equalize_passband_filter(observation_target ot, std::string_view h_resp_filepath, bland::ndarray::datatype dtype, bool validate) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    h = h.to(ot.device());
    return equalize_passband_filter(ot, h, validate);
}

cadence bliss::equalize_passband_filter(cadence ca, bland::ndarray h, bool validate) {
    for (auto &target : ca._observations) {
        target = equalize_passband_filter(target, h, validate);
    }
    return ca;
}

cadence bliss::equalize_passband_filter(cadence ca, std::string_view h_resp_filepath, bland::ndarray::datatype dtype, bool validate) {
    auto h = bland::read_from_file(h_resp_filepath, dtype);
    return equalize_passband_filter(ca, h, validate);
}
