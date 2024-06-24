#include "frequency_cpu_impl.hpp"

#include "internal/dispatcher.hpp"

#include "bland/ndarray.hpp"

#include <fmt/format.h>

using namespace bland;
using namespace bland::cpu;


ndarray bland::cpu::fft(ndarray a, ndarray &out) {
    fmt::print("WARN: cpu fft not implemented yet\n");
    return out;
}

ndarray bland::cpu::fft_shift_mag_square(ndarray a, ndarray &out) {

}