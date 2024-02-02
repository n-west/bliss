
#include <drift_search/filter_hits.hpp>
#include <core/flag_values.hpp>

using namespace bliss;

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
        if (remove_hit) {
            current_hit = hits.erase(current_hit);
        } else {
            ++current_hit;
        }
    }
    return hits;
}


scan bliss::filter_hits(scan scan_with_hits, filter_options options) {
    scan_with_hits.hits(filter_hits(scan_with_hits.hits(), options));
    return scan_with_hits;
}

observation_target bliss::filter_hits(observation_target scans_with_hits, filter_options options) {
    for (auto &scan : scans_with_hits._scans) {
        scan = filter_hits(scan, options);
    }
    return scans_with_hits;
}

cadence bliss::filter_hits(cadence cadence_with_hits, filter_options options) {
    for (auto &obs_target : cadence_with_hits._observations) {
        obs_target = filter_hits(obs_target, options);
    }
    return cadence_with_hits;
}
