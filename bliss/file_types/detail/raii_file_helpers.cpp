
#include "raii_file_helpers.hpp"

using namespace bliss::detail;

bliss::detail::raii_file_for_write::raii_file_for_write(std::string_view file_path) {
    _fd = open(file_path.data(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (_fd == -1) {
        auto str_err = strerror(errno);
        auto err_msg =
                fmt::format("write_hits_to_file: could not open file for writing (fd={}, error={})", _fd, str_err);
        throw std::runtime_error(err_msg);
    }
}

bliss::detail::raii_file_for_write::~raii_file_for_write() {
    close(_fd);
}


