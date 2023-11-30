
#include "file_types/h5_filterbank_file.hpp"

#include <bland/bland.hpp>
#include <iostream> // cerr
#include <vector>

using namespace bliss;

bliss::h5_filterbank_file::h5_filterbank_file(const std::string &file_path) {
    _h5_file_handle = H5::H5File(file_path, H5F_ACC_RDONLY);
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

    // if (std::get<2>(time_steps) < std::get<2>(freq_bins)) {
    //     spectrum_grid = row_major_f32array(std::get<0>(time_steps), std::get<0>(freq_bins));
    // } else {
    //     spectrum_grid = row_major_f32array(std::get<0>(freq_bins), std::get<0>(time_steps));
    //     spectrum_grid.transposeInPlace();
    // }

    // The data is read such that dim0 is time and dim1 is frequencyuj i/
    _h5_data_handle.read(spectrum_grid.data_ptr<float>(), H5::PredType::NATIVE_FLOAT);

    return spectrum_grid;
}

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
