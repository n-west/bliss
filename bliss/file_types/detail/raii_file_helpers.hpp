#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <errno.h>  // how to interpret errors from POSIX file
#include <fcntl.h>  // for 'open' and 'O_WRONLY'
#include <unistd.h> // for 'close'

namespace bliss {

namespace detail {
// Fairly dumb classes to wrap up error handling to one (two) places that handle the POSIX API
// details and use RAII to make sure we don't leak resources
struct raii_file_for_write {
    int _fd;
    raii_file_for_write(std::string_view file_path);

    ~raii_file_for_write();
};
struct raii_file_for_read {
    int _fd;
    raii_file_for_read(std::string_view file_path) {
        _fd = open(file_path.data(), O_RDONLY);
        if (_fd == -1) {
            auto str_err = strerror(errno);
            auto err_msg =
                    fmt::format("raii_file_for_read: could not open file ({}) for reading (fd={}, error={})", file_path, _fd, str_err);
            throw std::runtime_error(err_msg);
        }
    }

    ~raii_file_for_read() { close(_fd); }
};

}  // namespace detail

} // namespace bliss
