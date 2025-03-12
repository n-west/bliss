
#include "file_types/h5_filterbank_file.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <bland/bland.hpp>
#include <iostream> // cerr
#include <vector>
#include <array>

using namespace bliss;

constexpr bool pedantic = false;

// Default implementation for reading data_attr
template <typename T>
T bliss::h5_filterbank_file::read_data_attr(const std::string &key) {
    T val;
    if (_h5_data_handle.attrExists(key)) {
        auto attr  = _h5_data_handle.openAttribute(key);
        auto dtype = attr.getDataType();

        // Check the data type and perform the appropriate casting
        if (dtype == H5::PredType::NATIVE_INT16) {
            if constexpr (pedantic && !std::is_same_v<int16_t, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_INT16 in file\n", key, typeid(T).name());
            }
            int16_t val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        } else if (dtype == H5::PredType::NATIVE_UINT16) {
            if constexpr (pedantic && !std::is_same_v<uint16_t, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_UINT16 in file\n", key, typeid(T).name());
            }
            uint16_t val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        } else if (dtype == H5::PredType::NATIVE_INT32) {
            if constexpr (pedantic && !std::is_same_v<int32_t, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_INT32 in file\n", key, typeid(T).name());
            }
            int32_t val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        } else if (dtype == H5::PredType::NATIVE_UINT32) {
            if constexpr (pedantic && !std::is_same_v<uint32_t, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_UINT32 in file\n", key, typeid(T).name());
            }
            uint32_t val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        }  else if (dtype == H5::PredType::NATIVE_INT64) {
            if constexpr (pedantic && !std::is_same_v<int64_t, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_INT64 in file\n", key, typeid(T).name());
            }
            int64_t val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        }  else if (dtype == H5::PredType::NATIVE_UINT64) {
            if constexpr (pedantic && !std::is_same_v<uint64_t, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_UINT64 in file\n", key, typeid(T).name());
            }
            uint64_t val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        }  else if (dtype == H5::PredType::NATIVE_FLOAT) {
            if constexpr (pedantic && !std::is_same_v<float, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_FLOAT in file\n", key, typeid(T).name());
            }
            float val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        }  else if (dtype == H5::PredType::NATIVE_DOUBLE) {
            if constexpr (pedantic && !std::is_same_v<double, T>) {
                fmt::print("WARN: attr {} requested as {} but is NATIVE_DOUBLE in file\n", key, typeid(T).name());
            }
            double val;
            attr.read(dtype, &val);
            return static_cast<T>(val);
        } else {
            T val;
            attr.read(dtype, &val);
            return val;
        }
    } else {
        auto err_msg = fmt::format("H5 data does not have an attribute key {}", key);
        throw std::invalid_argument(err_msg);
    }
}

// Instantiated for common arithmetic types
template float       bliss::h5_filterbank_file::read_data_attr<float>(const std::string &key);
template double      bliss::h5_filterbank_file::read_data_attr<double>(const std::string &key);
template uint8_t     bliss::h5_filterbank_file::read_data_attr<uint8_t>(const std::string &key);
template uint16_t    bliss::h5_filterbank_file::read_data_attr<uint16_t>(const std::string &key);
template uint32_t    bliss::h5_filterbank_file::read_data_attr<uint32_t>(const std::string &key);
template uint64_t    bliss::h5_filterbank_file::read_data_attr<uint64_t>(const std::string &key);
template int8_t      bliss::h5_filterbank_file::read_data_attr<int8_t>(const std::string &key);
template int16_t     bliss::h5_filterbank_file::read_data_attr<int16_t>(const std::string &key);
template int32_t     bliss::h5_filterbank_file::read_data_attr<int32_t>(const std::string &key);
template int64_t     bliss::h5_filterbank_file::read_data_attr<int64_t>(const std::string &key);
// template std::string bliss::h5_filterbank_file::read_data_attr<std::string>(const std::string &key);

// Specialization for reading (byte) strings
template <>
std::string bliss::h5_filterbank_file::read_data_attr<std::string>(const std::string &key) {
    if (_h5_data_handle.attrExists(key)) {
        auto attr  = _h5_data_handle.openAttribute(key);
        auto dtype = attr.getDataType();

        // Check if the attribute is a byte string
        if (dtype.getClass() == H5T_STRING && !dtype.isVariableStr()) {
            // ATA pipeline emits fixed-length bytestrings for the source_name
            hsize_t size = attr.getInMemDataSize();
            std::vector<uint8_t> val(size);
            attr.read(dtype, val.data());

            // Convert the byte string to a std::string
            return std::string(val.begin(), val.end());
        } else if (dtype.getClass() == H5T_STRING && dtype.isVariableStr()) {
            // Some pipelines generate variable length bytestrings
            char* val;
            // hdf5 will allocate space for val
            attr.read(dtype, &val);

            std::string result(val);

            // Free the memory allocated by HDF5
            #if H5_VERSION_GE(1, 8, 13)
            H5free_memory(val);
            #else
            free(val);
            #endif

            return result;
        } else {
            auto err_msg = fmt::format("{} expected as string but is not a string type");
            throw std::invalid_argument(err_msg);
        }
    } else {
        auto err_msg = fmt::format("H5 data does not have an attribute key {}", key);
        throw std::invalid_argument(err_msg);
    }
}

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

    #if H5_VERSION_GE(1, 10, 1)
    unsigned int number_plugin_paths;
    std::string filter_paths{};
    auto h5_err = H5PLsize(&number_plugin_paths);
    for (unsigned int plugin_path_index=0; plugin_path_index < number_plugin_paths; ++plugin_path_index) {
        auto required_buffer_size = H5PLget(plugin_path_index, NULL, 0);
        std::string path_buffer;
        path_buffer.resize(required_buffer_size);
        required_buffer_size = H5PLget(plugin_path_index, path_buffer.data(), required_buffer_size);

        filter_paths += fmt::format("\t'{}'\n", path_buffer);
    }
    fmt::print("INFO: HDF5 looking for filter plugins in: \n{}", filter_paths);
    #endif

    _h5_file_handle = H5::H5File(file_path.data(), H5F_ACC_RDONLY);
    try {
        _h5_data_handle = _h5_file_handle.openDataSet("data");
    } catch (H5::FileIException h5_data_exception) {
        fmt::print("ERROR: h5_filterbank_file: got an exception while reading data. Cannot continue and rethrowing.\n");
        throw h5_data_exception;
    }

    try {
        // handle::nameExists is newer than hdf5 available from manylinux2014 centos
        if (H5Lexists(_h5_file_handle.getId(), "mask", H5P_DEFAULT) /*_h5_file_handle.nameExists("mask")*/) {
                _h5_mask_handle = _h5_file_handle.openDataSet("mask");
        } else {
            fmt::print("INFO: h5_filterbank_file: mask is not in this file. This is recoverable.\n");
        }
    } catch (H5::FileIException h5_mask_exception) {
        fmt::print("WARN: h5_filterbank_file: got an exception while reading mask. This is recoverable.\n");
    }

    try {
        auto h5_file_class = read_file_attr<std::string>("CLASS");
        if (h5_file_class != "FILTERBANK") {
            fmt::print("WARN: the h5 file has a 'CLASS' attribute that is *not* set to 'FILTERBANK' (is {}). We will assume this is compatible with a FILTERBANK and resume.\n", h5_file_class);
        }
    } catch (std::invalid_argument &e) {
        fmt::print("WARN: the h5 file does not have a 'CLASS' attribute. This should be set to FILTERBANK. Assuming this is a FILTERBANK file and resuming.\n");
    }

    // TODO: what is the support matrix we can handle?
    // We know most telescopes and archive data are 1.0 and GBT currently emits 2.0
    constexpr std::array<const char*, 2> supported_filterbank_versions = {"1.0", "2.0"};

    try {
        auto filterbank_version = read_file_attr<std::string>("VERSION");
        if (!std::any_of(supported_filterbank_versions.begin(),
                        supported_filterbank_versions.end(),
                        [filterbank_version](const char* supported_version) { return supported_version == filterbank_version; })) {

            auto warning = fmt::format("WARN: h5_filterbank_file: H5 FILTERBANK file VERSION field ({}) is not in known supported "
                        "versions list {}. Trying to read it anyway!\n",
                        filterbank_version,
                        supported_filterbank_versions);
            fmt::print(warning);
        }
    } catch (std::invalid_argument &e) {
        fmt::print("WARN: the h5 file does not have a 'VERSION' attribute to indicate what version of FILTERBANK this is. Will assume it is compatible and resume.\n");
    }

}


std::vector<int64_t> bliss::h5_filterbank_file::get_data_shape() {
    auto dataspace   = _h5_data_handle.getSpace();
    auto number_dims = dataspace.getSimpleExtentNdims();

    std::vector<hsize_t> dims(number_dims);
    dataspace.getSimpleExtentDims(dims.data());

    auto dim_labels = read_data_attr<std::vector<std::string>>("DIMENSION_LABELS");
    // tuples of size, initialized state, index. This will validate the dimensions
    // fit our expectations:
    // * there are 3 dimensions with unique labels of time, feed_id, frequency
    // and fix any known issues (like labels swapped)
    std::tuple<int64_t, bool, int> time_steps = {0, false, -1};
    std::tuple<int64_t, bool, int> feed_id = {0, false, -1};
    std::tuple<int64_t, bool, int> freq_bins  = {0, false, -1};
    for (int ii = 0; ii < dims.size(); ++ii) {
        if (!std::get<1>(time_steps) && dim_labels[ii] == "time") {
            time_steps = {dims[ii], true, ii};
        } else if (!std::get<1>(freq_bins) && dim_labels[ii] == "frequency") {
            freq_bins = {dims[ii], true, ii};
        } else if (!std::get<1>(feed_id) && dim_labels[ii] == "feed_id") {
            feed_id = {dims[ii], true, ii};
            if (dims[ii] != 1) {
                throw std::invalid_argument("Expected unity feed_id");
            }
        } else { // if there are more than 3 dims this will catch it
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
    if (std::get<1>(feed_id) == false) {
        throw std::invalid_argument("Could not find a feed_id dimension in dimension labels");
    }
    if (std::get<1>(freq_bins) == false) {
        throw std::invalid_argument("Could not find a frequency dimension in dimension labels");
    }
    if (std::get<0>(time_steps) == read_data_attr<int64_t>("nchans") && std::get<0>(freq_bins) != read_data_attr<int64_t>("nchans")) {
        static bool warning_issued = false;
        if (!warning_issued) {
            fmt::print("WARN h5_filterbank_file: the DIMENSION_LABELS appear out of order, time dimension matches nchans\n");
            warning_issued = true;
        }
        // There's some files that have the dim labels for time, channels swapped
        auto temp_time_steps = freq_bins;
        freq_bins            = time_steps;
        time_steps           = temp_time_steps;
    }

    auto shape = std::vector<int64_t>({std::get<0>(time_steps), std::get<0>(feed_id), std::get<0>(freq_bins)});
    return shape;
  }


bland::ndarray bliss::h5_filterbank_file::read_data(std::vector<int64_t> offset, std::vector<int64_t> count) {
    auto dataspace   = _h5_data_handle.getSpace();
    auto number_dims = dataspace.getSimpleExtentNdims();

    bland::ndarray spectrum_grid;

    auto shape = get_data_shape();

    if (offset.empty()) {
        offset = std::vector<int64_t>(shape.size(), 0);
    }
    if (count.empty()) {
        count = shape;
        count[0] -= offset[0];
        count[1] -= offset[1];
        count[2] -= offset[2];
    }
    // TODO: if we have cuda we should put this in unified memory since we *know* we'll move it to device
    // and it needs to pass through there anyway
    spectrum_grid = bland::ndarray(count, bland::ndarray::datatype::float32, bland::ndarray::dev::cpu);
    // TODO: validate both offset and count are size 3

    std::vector<hsize_t> offset_hsize(offset.begin(), offset.end());
    std::vector<hsize_t> count_hsize(count.begin(), count.end());

    dataspace.selectHyperslab(H5S_SELECT_SET, count_hsize.data(), offset_hsize.data());

    // Define the memory dataspace to receive the read data
    std::vector<hsize_t> grid_shape(count.begin(), count.end());
    H5::DataSpace        memspace(grid_shape.size(), grid_shape.data());

    // The row-major reading and axes we set up means frequency (most dense) is in last dim
    _h5_data_handle.read(spectrum_grid.data_ptr<float>(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);

    spectrum_grid = spectrum_grid.squeeze(1); // squeeze out the feed_id
    return spectrum_grid;
}

bland::ndarray bliss::h5_filterbank_file::read_mask(std::vector<int64_t> offset, std::vector<int64_t> count) {
    // if (_h5_mask_handle.has_value()) {
    //     auto h5_mask = _h5_mask_handle.value();
    //     auto dataspace   = h5_mask.getSpace();
    //     auto number_dims = dataspace.getSimpleExtentNdims();

    //     auto shape = get_data_shape();

    //     if (offset.empty()) {
    //         offset = std::vector<int64_t>(shape.size(), 0);
    //     }
    //     if (count.empty()) {
    //         count = shape;
    //         count[0] -= offset[0];
    //         count[1] -= offset[1];
    //         count[2] -= offset[2];
    //     }
    //     auto mask_grid = bland::ndarray(count, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);
    //     // TODO: validate both offset and count are size 3

    //     std::vector<hsize_t> offset_hsize(offset.begin(), offset.end());
    //     std::vector<hsize_t> count_hsize(count.begin(), count.end());

    //     dataspace.selectHyperslab(H5S_SELECT_SET, count_hsize.data(), offset_hsize.data());

    //     // Define the memory dataspace to receive the read data
    //     std::vector<hsize_t> grid_shape(count.begin(), count.end());
    //     H5::DataSpace        memspace(grid_shape.size(), grid_shape.data());

    //     // The row-major reading and axes we set up means frequency (most dense) is in last dim
    //     h5_mask.read(mask_grid.data_ptr<uint8_t>(), H5::PredType::NATIVE_UINT8, memspace, dataspace);

    //     mask_grid = mask_grid.squeeze(1); // squeeze out the feed_id
    //     return mask_grid;
    // } else {
        // The file has no "mask" dataset, it's typically zeros anyway so just allocate the appropriate number of uint8 zeros
        auto mask_grid = bland::zeros(count, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);
        mask_grid = mask_grid.squeeze(1); // squeeze out the feed_id
        return mask_grid;
    // }
}

std::string bliss::h5_filterbank_file::repr() {
    auto repr = fmt::format("File at {}\n    with CLASS {}, VERSION {}\n    has datasets:",
                            _h5_file_handle.getFileName(),
                            read_file_attr<std::string>("CLASS"),
                            read_file_attr<std::string>("VERSION"));

    // TODO, can we just discover number of datasets and figure out their dimensions to print?
    // _h5_file_handle.
    // fmt::format_to(repr, "    has datasets:\n");
    return repr;
}

std::string bliss::h5_filterbank_file::get_file_path() const {
    return _h5_file_handle.getFileName();
}
