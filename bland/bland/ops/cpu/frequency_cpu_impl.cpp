#include "frequency_cpu_impl.hpp"

#include "internal/dispatcher.hpp"

#include "bland/ndarray.hpp"

#include <fftw3.h>

#include <fmt/format.h>

using namespace bland;
using namespace bland::cpu;


ndarray bland::cpu::fft(ndarray x, ndarray &out) {
    fmt::print("WARN: cpu fft not implemented yet\n");
    return out;
}

ndarray bland::cpu::fft_shift_mag_square(ndarray x, ndarray &out) {
    auto shape = x.shape();
    int N = shape[0];
    int half_N = floor(N/2) + 1;

    auto in_ptr = x.data_ptr<float>();
    auto out_ptr = out.data_ptr<float>();

    fftwf_complex *fftdata = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * half_N);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, in_ptr, fftdata, FFTW_ESTIMATE);

    fftwf_execute(plan);

    int dc_index = ceil((N-1)/2.0f);
    // The R2C fft returns floor(N/2)+1 outputs. The N/2 are conjugates 
    for (int i = 0; i < half_N; ++i) {
        float real = fftdata[i][0];
        float imag = fftdata[i][1];
        int shift_index = dc_index + i;
        int rev_index = dc_index - i;
        auto mag_squared = real*real + imag*imag;
        out_ptr[shift_index] = mag_squared;
        if (i != 0) {
            out_ptr[rev_index] = mag_squared;
        }

    }

    fftwf_destroy_plan(plan);
    fftwf_free(fftdata);

    return out;
}