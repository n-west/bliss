#pragma once

#include <unistd.h>
#include <exception>
#include <fstream>

static int64_t get_fd_file_length(int fd) {
    off_t current_pos = lseek(fd, 0, SEEK_CUR); // Save current position
    off_t file_length = lseek(fd, 0, SEEK_END); // Seek to end of file
    lseek(fd, current_pos, SEEK_SET); // Restore original position

    if (file_length == -1) {
        throw std::runtime_error("Failed to get file length.");
    }

    return file_length;
}

static int64_t get_ifstream_file_length(std::ifstream &fstr) {
    // Save the current position
    std::streampos original_pos = fstr.tellg();

    // Seek to the end of the file
    fstr.seekg(0, std::ios::end);

    // Get the position, which is the size of the file
    std::streamsize size = fstr.tellg();

    // Seek back to the original position
    fstr.seekg(original_pos);

    return size;
}
