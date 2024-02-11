#pragma once

#include <core/integrate_drifts_options.hpp>

#include <bland/ndarray.hpp>

#include <tuple>

namespace bliss {

/**
 * linear-rounded integration kernel implemented with bland array primitives
 * 
 * Naive approach following a line through spectrum grid using
 * round-away-from-zero (commercial rounding) based on a slope of time span over
 * frequency span where the time span is always the full time extent and the
 * frequency span is the distance between the start and end of the linear drift.
 *
 * Note that if there are 8 time rows, the time span is 7. Likewise, the 8
 * drifts will have frequency spans of 0, 1, 2, 3, 4, 5, 6, 7 giving 8 slopes of
 * value 0/7, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7.
 */
[[nodiscard]] std::tuple<bland::ndarray, integrated_flags>
integrate_linear_rounded_bins(const bland::ndarray    &spectrum_grid,
                              const bland::ndarray    &rfi_mask,
                              integrate_drifts_options options);

bland::ndarray integrate_linear_rounded_bins(const bland::ndarray &spectrum_grid, integrate_drifts_options options);

} // namespace bliss
