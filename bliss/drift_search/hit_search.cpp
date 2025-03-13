
#include "bland/ndarray.hpp"
#include <drift_search/connected_components.hpp>

#include <drift_search/protohit_search.hpp>
#include <drift_search/hit_search.hpp>
#include <drift_search/local_maxima.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <cstdint>

using namespace bliss;



std::list<hit> bliss::hit_search(coarse_channel working_cc, hit_search_options options) {

    auto noise_estimate  = working_cc.noise_estimate();
    // We want to be able to iterate on an integrated_drift_plane and protohit search
    // integrated_drift_plane will throw (std::runtime_error) if there is no drift plane, we can catch this
    // and do some drift plane by default, which could be an iterative integrate + proto search
    

    std::vector<protohit> protohits;
    auto noise_estimate  = working_cc.noise_estimate();
    auto protohits = driftblock_protohit_search(working_cc, noise_estimate, options);
    auto dedrifted_plane = working_cc.integrated_drift_plane();
    protohits = protohit_search(dedrifted_plane, noise_estimate, options);
    // // We need to figure out some scoping issues, such as integration steps and drift_rate_info
    // if (options.iterative == true) {
    //     fmt::print("No drift plane found, generating one\n");
    //     // auto dedrifted_plane = dedrifted_scan.integrated_drift_plane();
    // } else {
    //     try {
    //     } catch (std::runtime_error &e) {
    //         fmt::print("ERROR: the dedrift plane does not exist");
    //     }
    // }

    // The rest of this is actually just translating protohits to hits

    auto integration_length = dedrifted_plane.integration_steps();
    auto drift_rate_info    = dedrifted_plane.drift_rate_info();

    std::list<hit> hits;
    for (const auto &c : protohits) {
        hit this_hit;
        this_hit.rate_index       = c.index_max.drift_index;
        this_hit.rfi_counts       = c.rfi_counts;
        this_hit.start_freq_index = c.index_max.frequency_channel;

        // Start frequency in Hz is bin * Hz/bin
        auto freq_offset        = working_cc.foff() * this_hit.start_freq_index;
        this_hit.start_freq_MHz = working_cc.fch1() + freq_offset;

        this_hit.drift_rate_Hz_per_sec = drift_rate_info[this_hit.rate_index].drift_rate_Hz_per_sec;

        auto signal_power = (c.max_integration - noise_estimate.noise_floor());

        // This is the unsmeared SNR
        this_hit.power = signal_power;
        this_hit.snr   = signal_power / c.desmeared_noise;

        this_hit.binwidth  = c.binwidth;
        this_hit.bandwidth = this_hit.binwidth * std::abs(1e6 * working_cc.foff());

        freq_offset             = working_cc.foff() * c.index_center.frequency_channel;
        this_hit.start_freq_MHz = working_cc.fch1() + freq_offset;
        this_hit.start_time_sec = working_cc.tstart() * 24 * 60 * 60; // convert MJD to seconds since MJ
        this_hit.duration_sec   = working_cc.tsamp() * integration_length;
        this_hit.integrated_channels = drift_rate_info[this_hit.rate_index].desmeared_bins * integration_length;
        this_hit.coarse_channel_number = working_cc._coarse_channel_number;
        this_hit.rfi_counts[flag_values::sigma_clip] = c.rfi_counts.at(flag_values::sigma_clip);
        this_hit.rfi_counts[flag_values::low_spectral_kurtosis] = c.rfi_counts.at(flag_values::low_spectral_kurtosis);
        this_hit.rfi_counts[flag_values::high_spectral_kurtosis] = c.rfi_counts.at(flag_values::high_spectral_kurtosis);
        hits.push_back(this_hit);
    }

    return hits;
}

scan bliss::hit_search(scan dedrifted_scan, hit_search_options options) {
    dedrifted_scan.add_coarse_channel_transform([options](coarse_channel cc) {
        auto hits = hit_search(cc, options);
        cc.set_hits(hits);
        return cc;
    });
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
