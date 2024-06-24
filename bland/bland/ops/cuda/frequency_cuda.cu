#include "frequency_cuda.cuh"

#include "internal/dispatcher.hpp"

#include "bland/ndarray.hpp"

#include <thrust/complex.h>
#include <complex>

#include <cufft.h>
#include <thrust/reverse.h>
#include <thrust/transform.h>

#include <fmt/core.h>

using namespace bland;
using namespace bland::cuda;

__global__ void rc2c_shift_abs_square(float* out, cufftComplex* fftdata, int N, int half_N) {
    // Read in the floor(N/2)+1 fftdata from a R2C fft, do an fftshift(square(abs(.)))
    // square(abs(.)) will be x * conj(x)
    // R2C only stores floor(N/2)+1 points and N/2 of those are symmetric but conjugated. Since we only care
    // about the magnitude, it doesn't matter but each read point will create two output points
    int dc_index = ceil((N-1)/2.0f);

    auto tid = threadIdx.x * blockIdx.x * blockDim.x;
    for (int n=tid; n < N/2; ++n) {
        auto val = fftdata[n];
        auto mag_squared = val.x*val.x + val.y*val.y;

        // Complete spectrum and do fftshift
        int shift_idx = dc_index+n;
        int rev_idx = dc_index-n;
        out[shift_idx] = mag_squared;
        if (n != 0) {
            out[rev_idx] = mag_squared;
        }
    }

}

ndarray bland::cuda::fft_shift_mag_square(ndarray x, ndarray &out) {
    cufftHandle plan;
    cufftCreate(&plan);

    auto shape = x.shape();
    int N = shape[0];
    int half_N = floor(N/2) + 1;

    constexpr int batch = 1;
    cufftPlan1d(&plan, N, CUFFT_R2C, batch);

    auto in_ptr = x.data_ptr<float>();
    auto out_ptr = out.data_ptr<float>();

    cufftComplex *fftdata;
    cudaMalloc(&fftdata, half_N * sizeof(cufftComplex));

    
    // The R2C fft returns floor(N/2)+1 outputs. The N/2 are conjugates 
    auto fft_res = cufftExecR2C(plan, in_ptr, fftdata);
    if (fft_res != CUFFT_SUCCESS) {
        fmt::print("ERR: CUFFT error: ExecR2C forward failed with {}\n", static_cast<int>(fft_res));
    }

    rc2c_shift_abs_square<<<1, 512>>>(out_ptr, fftdata, N, half_N);

    cudaFree(fftdata);
    cufftDestroy(plan);

    return out;
}

ndarray bland::cuda::fft(ndarray a, ndarray &out) {
    cufftHandle plan;
    cufftCreate(&plan);

    auto shape = a.shape();
    int N = shape[0];
    int half_N = floor(N/2) + 1;

    constexpr int batch = 1;
    cufftPlan1d(&plan, N, CUFFT_R2C, batch);

    auto idata = a.data_ptr<float>();
    auto odata = out.data_ptr<thrust::complex<float>>();
    
    // The R2C fft returns floor(N/2)+1 outputs
    auto fft_res = cufftExecR2C(plan, idata, reinterpret_cast<cufftComplex*>(odata));
    if (fft_res != CUFFT_SUCCESS) {
        fmt::print("ERR: CUFFT error: ExecR2C forward failed with {}\n", static_cast<int>(fft_res));
    }

    // Reverse and conjugate the first half of the FFT output, and store it in the second half
    thrust::transform(thrust::make_reverse_iterator(odata + half_N - 1),
                      thrust::make_reverse_iterator(odata + 1),
                      odata + half_N,
                      thrust::conj<float>);

    cufftDestroy(plan);

    return out;
}
