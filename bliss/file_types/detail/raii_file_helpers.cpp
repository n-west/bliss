
#include "raii_file_helpers.hpp"

#include <sys/stat.h>

using namespace bliss::detail;

bliss::detail::raii_file_for_write::raii_file_for_write(std::string_view file_path) {
    struct stat buffer;
    bool file_exists = (stat(file_path.data(), &buffer) == 0);
    if (file_exists) {
        fmt::print("WARN: writing to file {} that already exists. Existing contents will be overwritten.\n", file_path);
    }

    _fd = open(file_path.data(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (_fd == -1) {
        auto str_err = strerror(errno);
        auto err_msg =
                fmt::format("raii_file_for_write: could not open file ('{}') for writing (fd={}, error={})", file_path, _fd, str_err);
        throw std::runtime_error(err_msg);
    }
}

bliss::detail::raii_file_for_write::~raii_file_for_write() {
    close(_fd);
}


