
#include "file_types/events_file.hpp"

#include "hit.capnp.h"

#include "detail/cpnp_hit_builder.hpp"
#include "detail/raii_file_helpers.hpp"

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
        auto                        event_builder = message.initRoot<Event>();

        auto   hits_builder = event_builder.initHits(this_event.hits.size());
        size_t hit_index    = 0;
        for (auto this_hit : this_event.hits) {
            auto signal_builder = hits_builder[hit_index].initSignal();
            hits_builder[hit_index].initFilterbank();
            detail::bliss_hit_to_capnp_signal_message(signal_builder, this_hit);
            hit_index += 1;
        }

        capnp::writeMessageToFd(out_file._fd, message);
    }
}

std::vector<event> bliss::read_events_from_file(std::string_view file_path) {

    std::vector<event> events;

    auto in_file = detail::raii_file_for_read(file_path);

    while (true) {

        try {
            capnp::StreamFdMessageReader message(in_file._fd);

            auto event_reader      = message.getRoot<Event>();
            auto deserialized_hits = event_reader.getHits();
            // deserialized_event
            event new_event;
            for (auto deser_hit = deserialized_hits.begin(); deser_hit != deserialized_hits.end(); ++deser_hit) {
                auto signal_reader = deser_hit->getSignal();
                hit  new_hit       = detail::capnp_signal_message_to_bliss_hit(signal_reader);
                new_event.hits.push_back(new_hit);
            }
            events.push_back(new_event);
        } catch (kj::Exception &e) {
            // We've reached the end of the file.
            break;
        }
    }

    return events;
}
