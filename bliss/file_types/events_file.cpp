
#include "file_types/events_file.hpp"

#include "hit.capnp.h"

#include "detail/raii_file_helpers.hpp"
#include "detail/cpnp_hit_builder.hpp"

#include <stdexcept>

#include <capnp/c++.capnp.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/exception.h>

#include <fmt/core.h>
#include <fmt/format.h>

using namespace bliss;

void bliss::write_events_to_file(std::vector<event> events, std::string_view file_path) {

    auto out_file = detail::raii_file_for_write(file_path);
    for (size_t event_index = 0; event_index < events.size(); ++event_index) {
        auto                        this_event = events[event_index];
        capnp::MallocMessageBuilder message;
        auto                        event_builder    = message.initRoot<Event>();

        auto hits_builder = event_builder.initHits(this_event.hits.size());
        size_t hit_index = 0;
        for (auto this_hit : this_event.hits) {
            auto signal_builder = hits_builder[hit_index].initSignal();
            detail::bliss_hit_to_capnp_signal_message(signal_builder, this_hit);
        }

        capnp::writeMessageToFd(out_file._fd, message);
    }
}

std::vector<event> bliss::read_events_from_file(std::string_view base_filename) {

}
