
#include "file_types/h5_filterbank_file.hpp"

#include <bland/bland.hpp>
#include <iostream> // cerr
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/format.h>

using namespace bliss;

// Default implementation for reading data_attr
template <typename T>
T bliss::h5_filterbank_file::read_data_attr(const std::string &key) {
  T val;
  if (_h5_data_handle.attrExists(key)) {
    auto attr  = _h5_data_handle.openAttribute(key);
    auto dtype = attr.getDataType();

    attr.read(dtype, &val);
    return val;
  } else {
    auto err_msg = fmt::format("H5 data does not have an attribute key {}", key);
    throw std::invalid_argument(err_msg);
  }
}

// Instantiated for common arithmetic types
template float bliss::h5_filterbank_file::read_data_attr<float>(const std::string &key);
template double bliss::h5_filterbank_file::read_data_attr<double>(const std::string &key);
template uint8_t bliss::h5_filterbank_file::read_data_attr<uint8_t>(const std::string &key);
template uint16_t bliss::h5_filterbank_file::read_data_attr<uint16_t>(const std::string &key);
template uint32_t bliss::h5_filterbank_file::read_data_attr<uint32_t>(const std::string &key);
template uint64_t bliss::h5_filterbank_file::read_data_attr<uint64_t>(const std::string &key);
template int8_t bliss::h5_filterbank_file::read_data_attr<int8_t>(const std::string &key);
template int16_t bliss::h5_filterbank_file::read_data_attr<int16_t>(const std::string &key);
template int32_t bliss::h5_filterbank_file::read_data_attr<int32_t>(const std::string &key);
template int64_t bliss::h5_filterbank_file::read_data_attr<int64_t>(const std::string &key);
template std::string bliss::h5_filterbank_file::read_data_attr<std::string>(const std::string &key);

// Specialized for vector<string>
template <>
std::vector<std::string> bliss::h5_filterbank_file::read_data_attr<std::vector<std::string>>(const std::string &key) {
    if (_h5_data_handle.attrExists(key)) {
        H5::Attribute             attr = _h5_data_handle.openAttribute(key);
        std::vector<const char *> vals_as_read_from_h5(attr.getInMemDataSize() / sizeof(char *));
        attr.read(attr.getDataType(), vals_as_read_from_h5.data());

        std::vector<std::string> vals;
        for (size_t i = 0; i < vals_as_read_from_h5.size(); ++i) {
            vals.emplace_back(vals_as_read_from_h5[i]);
        }

        return vals;
    } else {
        throw std::invalid_argument("H5 data does not have an attribute key");
    }
}


bliss::h5_filterbank_file::h5_filterbank_file(std::string_view file_path) {
    _h5_file_handle = H5::H5File(file_path.data(), H5F_ACC_RDONLY);
    _h5_data_handle = _h5_file_handle.openDataSet("data");
    _h5_mask_handle = _h5_file_handle.openDataSet("mask");

    if (read_file_attr<std::string>("CLASS") != "FILTERBANK") {
        throw std::invalid_argument("H5 file CLASS is not FILTERBANK");
    }

    // TODO: what is the support matrix we can handle?
    if (read_file_attr<std::string>("VERSION") != "1.0") {
        throw std::invalid_argument("H5 file VERSION is not 1.0");
    }
}

bland::ndarray bliss::h5_filterbank_file::read_data() {
    auto space       = _h5_data_handle.getSpace();
    auto number_dims = space.getSimpleExtentNdims();

    std::vector<hsize_t> dims(number_dims);
    space.getSimpleExtentDims(dims.data());

    auto dim_labels = read_data_attr<std::vector<std::string>>("DIMENSION_LABELS");
    // tuples of size, initialized state, index
    std::tuple<int64_t, bool, int> time_steps = {0, false, -1};
    std::tuple<int64_t, bool, int> freq_bins  = {0, false, -1};
    for (int ii = 0; ii < dims.size(); ++ii) {
        if (dim_labels[ii] == "time") {
            time_steps = {dims[ii], true, ii};
        } else if (dim_labels[ii] == "frequency") {
            freq_bins = {dims[ii], true, ii};
        } else if (dim_labels[ii] == "feed_id") {
            if (dims[ii] != 1) {
                throw std::invalid_argument("Expected unity feed_id");
            }
        } else {
            // unknown dimension. If it's size 1, we're probably OK though
            if (dims[ii] != 1) {
                // TODO, we're using fmtlib now, update this. Does fmtlib act as a proper
                // "logging" lib?
                std::cerr << "Got unknown " << ii << " dimension: " << dim_labels[ii]
                          << ". Continuing since it is size 1." << std::endl;
            } else {
                throw std::invalid_argument("Unknown dimension of non-unity size in data");
            }
        }
    }
    if (std::get<1>(time_steps) == false) {
        throw std::invalid_argument("Could not find a time dimension in dimension labels");
    }
    if (std::get<1>(freq_bins) == false) {
        throw std::invalid_argument("Could not find a frequency dimension in dimension labels");
    }

    bland::ndarray spectrum_grid({std::get<0>(time_steps), std::get<0>(freq_bins)});

    // The data is read such that dim0 is time and dim1 is frequencyuj i/
    _h5_data_handle.read(spectrum_grid.data_ptr<float>(), H5::PredType::NATIVE_FLOAT);

    return spectrum_grid;
}



bland::ndarray bliss::h5_filterbank_file::read_mask() {
    auto space       = _h5_mask_handle.getSpace();
    auto number_dims = space.getSimpleExtentNdims();

    std::vector<hsize_t> dims(number_dims);
    space.getSimpleExtentDims(dims.data());

    auto dim_labels = read_data_attr<std::vector<std::string>>("DIMENSION_LABELS");
    // tuples of size, initialized state, index
    std::tuple<int64_t, bool, int> time_steps = {0, false, -1};
    std::tuple<int64_t, bool, int> freq_bins  = {0, false, -1};
    for (int ii = 0; ii < dims.size(); ++ii) {
        if (dim_labels[ii] == "time") {
            time_steps = {dims[ii], true, ii};
        } else if (dim_labels[ii] == "frequency") {
            freq_bins = {dims[ii], true, ii};
        } else if (dim_labels[ii] == "feed_id") {
            if (dims[ii] != 1) {
                throw std::invalid_argument("Expected unity feed_id");
            }
        } else {
            // unknown dimension. If it's size 1, we're probably OK though
            if (dims[ii] != 1) {
                // TODO, make fmt print to cerr
                fmt::print("Got unknown {} dimension: {}. Continuing since it is size 1.\n", ii, dim_labels[ii]);
            } else {
                throw std::invalid_argument("Unknown dimension of non-unity size in data");
            }
        }
    }
    if (std::get<1>(time_steps) == false) {
        throw std::invalid_argument("Could not find a time dimension in dimension labels");
    }
    if (std::get<1>(freq_bins) == false) {
        throw std::invalid_argument("Could not find a frequency dimension in dimension labels");
    }

    bland::ndarray mask_grid({std::get<0>(time_steps), std::get<0>(freq_bins)}, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

    // The data is read such that dim0 is time and dim1 is frequency
    _h5_mask_handle.read(mask_grid.data_ptr<uint8_t>(), H5::PredType::NATIVE_UINT8);

    return mask_grid;
}

std::string bliss::h5_filterbank_file::repr() {
    auto repr = fmt::format("File at {}\n    with CLASS {}, VERSION {}\n    has datasets:", _h5_file_handle.getFileName(), read_file_attr<std::string>("CLASS"), read_file_attr<std::string>("VERSION"));

    // TODO, can we just discover number of datasets and figure out their dimensions to print?
    // _h5_file_handle.    
    // fmt::format_to(repr, "    has datasets:\n");
    return repr;
}
