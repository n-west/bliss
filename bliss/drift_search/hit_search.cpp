
#include "bland/ndarray.hpp"
#include <drift_search/connected_components.hpp>

#include <drift_search/protohit_search.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/local_maxima.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <cstdint>

using namespace bliss;



std::list<hit> bliss::hit_search(coarse_channel dedrifted_scan, hit_search_options options) {

    auto protohits = protohit_search(dedrifted_scan, options);

    // We have to be on cpu for now
    dedrifted_scan.set_device("cpu");
    dedrifted_scan.push_device();

    auto noise_stats        = dedrifted_scan.noise_estimate();
    auto dedrifted_plane    = dedrifted_scan.integrated_drift_plane();
    auto drift_rate_info    = dedrifted_plane.drift_rate_info();
    auto integration_length = dedrifted_plane.integration_steps();

    std::list<hit> hits;
    for (const auto &c : protohits) {
        hit this_hit;
        this_hit.rate_index       = c.index_max.drift_index;
        this_hit.rfi_counts       = c.rfi_counts;
        this_hit.start_freq_index = c.index_max.frequency_channel;

        // Start frequency in Hz is bin * Hz/bin
        auto freq_offset        = dedrifted_scan.foff() * this_hit.start_freq_index;
        this_hit.start_freq_MHz = dedrifted_scan.fch1() + freq_offset;

        this_hit.drift_rate_Hz_per_sec = drift_rate_info[this_hit.rate_index].drift_rate_Hz_per_sec;

        auto signal_power = (c.max_integration - noise_stats.noise_floor());

        // This is the unsmeared SNR
        this_hit.power = signal_power;
        this_hit.snr   = signal_power / c.desmeared_noise;

        this_hit.binwidth  = c.binwidth;
        this_hit.bandwidth = this_hit.binwidth * std::abs(1e6 * dedrifted_scan.foff());

        freq_offset             = dedrifted_scan.foff() * c.index_center.frequency_channel;
        this_hit.start_freq_MHz = dedrifted_scan.fch1() + freq_offset;
        this_hit.start_time_sec = dedrifted_scan.tstart() * 24 * 60 * 60; // convert MJD to seconds since MJ
        this_hit.duration_sec   = dedrifted_scan.tsamp() * integration_length;
        hits.push_back(this_hit);
    }

    return hits;
}

// TODO: defer execution of hit search until hits from a specific cc are requested
scan bliss::hit_search(scan dedrifted_scan, hit_search_options options) {
    auto number_coarse_channels = dedrifted_scan.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        // We need a copy of this coarse_channel so that the current device settings
        // in the channel stick around even if something downstream changes the device
        // before we process this function.
        auto cc = dedrifted_scan.read_coarse_channel(cc_index);
        auto cc_copy = std::make_shared<coarse_channel>(*cc);
        auto find_coarse_channel_hits_func = [cc_copy, options]() {
            auto hits = hit_search(*cc_copy, options);
            if (options.detach_graph) {
                cc_copy->detach_drift_plane();
            }
            return hits;
        };
        cc->add_hits(find_coarse_channel_hits_func);
    }
    return dedrifted_scan;
}

observation_target bliss::hit_search(observation_target dedrifted_target, hit_search_options options) {
    for (auto &dedrifted_scan : dedrifted_target._scans) {
        dedrifted_scan = hit_search(dedrifted_scan, options);
    }
    return dedrifted_target;
}

cadence bliss::hit_search(cadence dedrifted_cadence, hit_search_options options) {
    for (auto &obs_target : dedrifted_cadence._observations) {
        obs_target = hit_search(obs_target, options);
    }
    return dedrifted_cadence;
}
