#pragma once

#include <core/integrate_drifts_options.hpp>
#include <core/frequency_drift_plane.hpp>

#include <bland/ndarray.hpp>

#include <tuple>

namespace bliss {

/**
 * linear-rounded integration kernel implemented for cpu
*/
[[nodiscard]] frequency_drift_plane integrate_linear_rounded_bins_cuda(bland::ndarray           spectrum_grid,
                                                                       bland::ndarray           rfi_mask,
                                                                       integrate_drifts_options options);

bland::ndarray integrate_linear_rounded_bins_cuda(bland::ndarray spectrum_grid, integrate_drifts_options options);

} // namespace bliss
