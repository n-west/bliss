
#include "core/scan.hpp"
#include <fmt/format.h>
#include <stdexcept>

using namespace bliss;

bliss::scan::scan(const filterbank_data &fb_data) : filterbank_data(fb_data) {}

// bliss::scan::scan(const scan &other) : filterbank_data(other) {}

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
    int            number_coarse_channels = get_number_coarse_channels();
    for (int cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = get_coarse_channel(cc_index);
        try {
            auto this_channel_hits = cc->hits();
            all_hits.insert(all_hits.end(), this_channel_hits.cbegin(), this_channel_hits.cend());
        } catch (const std::bad_optional_access &e) {
            fmt::print("no hits available from coarse channel {}\n", cc_index);
            // TODO: catch specific exception we know might be thrown
        }
    }
    return all_hits;
}

bliss::scan bliss::scan::extract_coarse_channels(int start_channel, int count) {
    auto sliced_scan = *this;

    // what are implications of negative numbers here?
    sliced_scan._coarse_channel_offset += start_channel;
    sliced_scan._num_coarse_channels = count;

    int fine_channels_per_coarse = std::get<0>(_inferred_channelization);

    sliced_scan._fch1 = _fch1 + _foff * fine_channels_per_coarse * start_channel;

    for (int channel_index = start_channel; channel_index < start_channel + count; ++channel_index) {
        // This probably isn't really necessary since it's done behind the scenes
        // _coarse_channels.insert({channel_index, get_coarse_channel(channel_index)});
    }

    return sliced_scan;
}
