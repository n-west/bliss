
#include "core/scan.hpp"
#include <stdexcept>
#include <fmt/format.h>

using namespace bliss;


bliss::scan::scan(const filterbank_data &fb_data) : filterbank_data(fb_data) {}

// scan::state_tuple bliss::scan::get_state() {
//     std::list<hit> hits_state = _hits.value_or(std::list<hit>{});
//     noise_stats::state_tuple noise_state;
//     if (_noise_stats.has_value()) {
//         noise_state = _noise_stats.value().get_state();
//     }
//     auto state = std::make_tuple(
//         filterbank_data::get_state(),
//         bland::ndarray() /*doppler_spectrum*/,
//         _integration_length.value_or(0),
//         hits_state,
//         noise_state
//     );
//     return state;
// }

std::list<hit> bliss::scan::hits() {
    std::list<hit> all_hits;
    int number_coarse_channels = get_number_coarse_channels();
    for (int cc_index=0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = get_coarse_channel(cc_index);
        try {
            auto this_channel_hits = cc->hits();
            all_hits.insert(all_hits.end(), this_channel_hits.cbegin(), this_channel_hits.cend());
        } catch (std::exception &e) {
            fmt::print("no hits available from coarse channel {}\n", cc_index);
            // TODO: catch specific exception we know might be thrown
        }
    }
    return all_hits;
}

