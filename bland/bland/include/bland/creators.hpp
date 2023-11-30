#pragma once

#include "ndarray.hpp"

namespace bland {

template <typename START_TYPE, typename END_TYPE, typename STEP_TYPE>
ndarray arange(START_TYPE start, END_TYPE end, STEP_TYPE step, DLDataType dtype, DLDevice device = default_device);

/**
 * Generate a 1d array with evenly spaced values within the given interval.
 * arange without an explicit datatype will infer datatype from the types of start, end, step
 *
 */
template <typename START_TYPE, typename END_TYPE, typename STEP_TYPE>
ndarray arange(START_TYPE start, END_TYPE end, STEP_TYPE step = 1, DLDevice device = default_device);

/**
 * Generate `number_steps` equally spaced items between `start` and `end`
 */
template <typename T>
ndarray linspace(T start, T end, size_t number_steps, DLDataType dtype, DLDevice device = default_device);

/**
 * Generate a 1d array with evenly spaced values within the given interval.
 * linspace without an explicit datatype will infer datatype from the types of start and end
 *
 */
template <typename T>
ndarray linspace(T start, T end, size_t number_steps, DLDevice device = default_device);

ndarray zeros(std::vector<int64_t> shape, DLDataType dtype = default_dtype, DLDevice device = default_device);

ndarray ones(std::vector<int64_t> shape, DLDataType dtype = default_dtype, DLDevice device = default_device);

template <typename T>
ndarray rand_normal(std::vector<int64_t> shape, T mean = 0, T stddev = 1, DLDataType dtype = default_dtype, DLDevice device = default_device);

template <typename T>
ndarray rand_uniform(std::vector<int64_t> shape, T low=0, T high=1, DLDataType dtype = default_dtype, DLDevice device = default_device);

} // namespace bland
