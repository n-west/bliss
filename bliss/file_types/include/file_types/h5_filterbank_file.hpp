#pragma once

// #include "convenience/datatypedefs.hpp"
#include <bland/bland.hpp>
#include <H5Cpp.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bliss {

/**
 * Read an H5 file, parse attributes, and make underlying data available
 *
 * This is specific to H5 filterbank files, so there's a few assumptions:
 *
 * This is based on the CLASS FILTERBANK with VERSION 1.0. This contains two H5 datasets:
 * `data` and `mask` that are readable.
 *
 * The dataset metadata fields are all optional for now until I can find a spec that
 * indicates what it should really be.
 */
class h5_filterbank_file
{
public:
  h5_filterbank_file(const std::string &file_path);

  template <typename T>
  T read_file_attr(const std::string &key);

  template <typename T>
  T read_data_attr(const std::string &key);

  bland::ndarray read_data();

  bland::ndarray read_mask();

private:
  H5::H5File  _h5_file_handle;
  H5::DataSet _h5_data_handle;
  H5::DataSet _h5_mask_handle;
};

template <typename T>
T bliss::h5_filterbank_file::read_file_attr(const std::string &key)
{

  T val;
  if (_h5_file_handle.attrExists(key)) {
    auto attr  = _h5_file_handle.openAttribute(key);
    auto dtype = attr.getDataType();
    attr.read(dtype, val);
    return val;
  } else {
    throw std::invalid_argument("H5 file does not have an attribute key");
  }
}

template <typename T>
T bliss::h5_filterbank_file::read_data_attr(const std::string &key)
{

  T val;
  if (_h5_data_handle.attrExists(key)) {
    auto attr  = _h5_data_handle.openAttribute(key);
    auto dtype = attr.getDataType();

    attr.read(dtype, &val);
    return val;
  } else {
    throw std::invalid_argument("H5 file does not have an attribute key");
  }
}

template <>
std::vector<std::string> bliss::h5_filterbank_file::read_data_attr<std::vector<std::string>>(const std::string &key);

} // namespace bliss