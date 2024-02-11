#pragma once

#include <core/integrate_drifts_options.hpp>

#include <bland/ndarray.hpp>

#include <tuple>

namespace bliss {

/**
 * linear-rounded integration kernel implemented for cpu
*/
constexpr bool collect_rfi = true;
[[nodiscard]] std::tuple<bland::ndarray, integrated_flags>
integrate_linear_rounded_bins_cpu(const bland::ndarray    &spectrum_grid,
                                  const bland::ndarray    &rfi_mask,
                                  integrate_drifts_options options);

bland::ndarray integrate_linear_rounded_bins_cpu(const bland::ndarray &spectrum_grid, integrate_drifts_options options);

} // namespace bliss
