
#include <core/event.hpp>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

using namespace bliss;

std::string bliss::event::repr() {

    std::vector<std::string> hit_reprs;
    for (auto &hit : hits) {
        hit_reprs.push_back(hit.repr());
    }
    auto r = fmt::format("event: .starting_frequency_Hz={:.0f} .average_drift_rate_Hz_per_sec={:.2f}"
                         ".average_power={} .average_snr={} .event_start_seconds={} .event_end_seconds={}\n"
                         ".hits=[{}]",
                         starting_frequency_Hz,
                         average_drift_rate_Hz_per_sec,
                         average_power,
                         average_snr,
                         event_start_seconds,
                         event_end_seconds,
                         hit_reprs);
    return r;
}