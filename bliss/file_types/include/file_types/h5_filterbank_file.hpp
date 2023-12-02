#pragma once

// #include "convenience/datatypedefs.hpp"
#include <bland/bland.hpp>
#include <H5Cpp.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * \brief Where does this go?
*/

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

  /**
   * Read an HDF5 file-scoped attribute with the given key. The return type of the value is given by
   * template parameter `T`.
   * 
   * WARNING: An incorrect template type will result in unexpected return value
   * since the type is not checked to match the template parameter.
  */
  template <typename T>
  T read_file_attr(const std::string &key);

  /**
   * Read an HDF5 dataset-scoped attribute of the `data` dataset with the given key. The return type of the value is given by
   * template parameter `T`.
   * 
   * WARNING: An incorrect template type will result in unexpected return value
   * since the type is not checked to match the template parameter.
  */
  template <typename T>
  T read_data_attr(const std::string &key);

  /**
   * Read the `data` dataset to a new ndarray
  */
  bland::ndarray read_data();

  /**
   * Read the `mask` dataset to a new ndarray
  */
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