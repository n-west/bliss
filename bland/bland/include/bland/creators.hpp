#pragma once

#include "ndarray.hpp"

namespace bland {

/**
 * Create a 1d array with evenly spaced values between [start, end).
 * arange with an explicit datatype will cast the start, end, step values to
 * the provided dtype before creating the array.
 */
template <typename START_TYPE, typename END_TYPE, typename STEP_TYPE>
ndarray arange(START_TYPE start, END_TYPE end, STEP_TYPE step, DLDataType dtype, DLDevice device = default_device);

/**
 * Create a 1d array with evenly spaced values between [start, end).
 * arange without an explicit datatype will infer datatype from the types of start, end, step which must
 * all match.
 */
template <typename START_TYPE, typename END_TYPE, typename STEP_TYPE>
ndarray arange(START_TYPE start, END_TYPE end, STEP_TYPE step = 1, DLDevice device = default_device);

/**
 * Create an array of `number_steps` equally spaced items between `start` and `end`
 */
template <typename T>
ndarray linspace(T start, T end, size_t number_steps, DLDataType dtype, DLDevice device = default_device);

/**
 * Create a 1d array with evenly spaced values within the given interval.
 * linspace without an explicit datatype will infer datatype from the types of start and end
 *
 */
template <typename T>
ndarray linspace(T start, T end, size_t number_steps, DLDevice device = default_device);

/**
 * Return an array of the given shape, dtype, device with all data initialized to 0.
*/
ndarray zeros(std::vector<int64_t> shape, DLDataType dtype = default_dtype, DLDevice device = default_device);

/**
 * Return an array of the given shape, dtype, device with all data initialized to 1.
*/
ndarray ones(std::vector<int64_t> shape, DLDataType dtype = default_dtype, DLDevice device = default_device);

/**
 * Return an array of the given shape, dtype, device with all data initialized by sampling a normal distribution. The
 * mean and stddev are optional.
 * 
 * The distribution is N(\mu, \sigma) which is equivalent to N(0,1) * stddev + mu.
 * 
 * Only valid for float and double dtypes. meand and stddev are cast to the provided dtype (default float)
*/
template <typename T>
ndarray rand_normal(std::vector<int64_t> shape, T mean = 0, T stddev = 1, DLDataType dtype = default_dtype, DLDevice device = default_device);

/**
 * Return an array of the given shape, dtype, device with all data initialized using a uniform distribution between (low, high]. The
 * datatype is used to cast low and high before sampling from the distribution.
*/
template <typename T>
ndarray rand_uniform(std::vector<int64_t> shape, T low=0, T high=1, DLDataType dtype = default_dtype, DLDevice device = default_device);

} // namespace bland
