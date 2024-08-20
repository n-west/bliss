
#include <drift_search/filter_hits.hpp>
#include <core/flag_values.hpp>

#include <fmt/format.h>

#include <stdexcept>

using namespace bliss;

constexpr double eps = 1e-6;

std::list<hit> bliss::filter_hits(std::list<hit> hits, filter_options options) {

    auto current_hit = hits.begin();
    while (current_hit != hits.end()) {
        bool remove_hit = false;
        // TODO: logic to decide if we should remove a hit
        if (current_hit->rfi_counts[flag_values::filter_rolloff] > 0) {
            remove_hit = true;
        }
        if (current_hit->rfi_counts[flag_values::low_spectral_kurtosis] > 0) {
            remove_hit = true;
        }
        if (options.filter_zero_drift) {
            // With floating-point consider any drift rate < eps
            if (std::fabs(current_hit->drift_rate_Hz_per_sec - 0) < eps) {
                remove_hit = true;
            }
        }
        // if (current_hit->rfi_counts[flag_values::high_spectral_kurtosis < 0) {
        //     remove_hit = true;
        // }
        if (remove_hit) {
            current_hit = hits.erase(current_hit);
        } else {
            ++current_hit;
        }
    }
    return hits;
}

coarse_channel bliss::filter_hits(coarse_channel cc_with_hits, filter_options options) {
    if (cc_with_hits.has_hits()) {
        auto original_hits = cc_with_hits.hits();
        auto filtered_hits = filter_hits(original_hits, options);
        cc_with_hits.set_hits(filtered_hits);
        return cc_with_hits;
    } else {
        throw std::invalid_argument("coarse channel has no hits");
    }
}


scan bliss::filter_hits(scan scan_with_hits, filter_options options) {
    // scan_with_hits.hits(filter_hits(scan_with_hits.hits(), options));
    // iterate over all of the coarse channels, calling filter_hits per coarse channel
    int number_coarse_channels = scan_with_hits.get_number_coarse_channels();
    for (int cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = scan_with_hits.peak_coarse_channel(cc_index);
        if (cc != nullptr) {
            try {
                *cc = filter_hits(*cc, options);
            } catch (const std::logic_error &e) {
                fmt::print("WARN: caught exception ({}) while filtering hits from pipeline on coarse channel {}, this might indicate a bad pipeline that didn't produce hits\n", e.what(), cc_index);
            }
        }
    }
    return scan_with_hits;
}

observation_target bliss::filter_hits(observation_target observation_with_hits, filter_options options) {
    for (auto &scan : observation_with_hits._scans) {
        scan = filter_hits(scan, options);
    }
    return observation_with_hits;
}

cadence bliss::filter_hits(cadence cadence_with_hits, filter_options options) {
    for (auto &obs_target : cadence_with_hits._observations) {
        obs_target = filter_hits(obs_target, options);
    }
    return cadence_with_hits;
}
