
#if BLISS_USE_CAPNP

#include "file_types/hits_file.hpp"

#include <fcntl.h>  // for 'open' and 'O_WRONLY'
#include <unistd.h> // for 'close'

#include <stdexcept>

#include <capnp/c++.capnp.h>
#include <capnp/message.h>
#include <capnp/serialize.h>

using namespace bliss;

void bliss::write_hits_to_file(Hit hit, std::string_view file_path) {

    int fd = open(file_path.data(), O_WRONLY | O_CREAT);

    if (fd == -1) {
        throw std::runtime_error("write_hits_to_file: could not open file for writing");
    }

    capnp::MallocMessageBuilder message;

    auto event_builder = message.initRoot<Event>();
    auto hits_builder = event_builder.initHits(3); // Requires a size now, we'd have 3 ONs...

    // A hit is found in a single filterbank
    auto hit_builder = message.initRoot<Hit>();

    auto signal_builder = hit_builder.initSignal();
    signal_builder.setFrequency(0);
    signal_builder.setDriftRate(0);
    signal_builder.setDriftRate(0);

    auto filterbank_builder = hit_builder.initFilterbank();
    // TODO: fill in all of the details here

    capnp::writeMessageToFd(fd, message);

    // Close the file.
    close(fd);
}

#endif // BLISS_USE_CAPNP

