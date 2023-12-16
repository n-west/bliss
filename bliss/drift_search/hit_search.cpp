
#include "bland/ndarray.hpp"
#include <drift_search/hit_search.hpp>

#include <fmt/core.h>
#include <cstdint>
#include <queue>

using namespace bliss;

float compute_signal_threshold(const noise_stats &noise_stats, int64_t integration_length, float snr_threshold) {
    // When the signal amplitude is snr_threshold above the noise floor, we have a 'prehit' (a signal that naively
    // passes a hard threshold) that is when S/N > snr_threshold Given a noise floor estimate of nf, signal amplitude s,
    // noise amplitude n...
    // S = (s - nf)**2
    // N = (n)**2         our estimate has already taken in to account noise floor
    // (s-nf)/(n) > sqrt(snr_threshold)
    // s-nf > n * sqrt(snr_threshold)
    // s > nf + sqrt(N * snr_threshold)
    // Since the noise power was estimate before integration, it also decreases by sqrt of integration length
    float integration_adjusted_noise_power = noise_stats.noise_power() / std::sqrt(integration_length);
    auto  threshold = noise_stats.noise_floor() + std::sqrt(integration_adjusted_noise_power * snr_threshold);
    return threshold;
}

bland::ndarray bliss::hard_threshold_drifts(const bland::ndarray &dedrifted_spectrum,
                                            const noise_stats    &noise_stats,
                                            int64_t               integration_length,
                                            float                 snr_threshold) {

    auto hard_threshold = compute_signal_threshold(noise_stats, integration_length, snr_threshold);

    auto threshold_mask = dedrifted_spectrum > hard_threshold;

    return threshold_mask;
}

// Helper to abstract out incremem
struct stride_helper {

    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    stride_helper(std::vector<int64_t> shape, std::vector<int64_t> stride) : shape(shape), stride(stride) {}

    int64_t to_linear_offset(const nd_coords &coords) const {
        // TODO compute linear index of curr_coords from shape and stride
        auto linear_offset = 0;
        for (int dim = 0; dim < shape.size(); ++dim) {
            linear_offset += coords[dim] * stride[dim]; // TODO: also add offset
        }
        return linear_offset;
    }
};

std::vector<component> bliss::find_components_in_binary_mask(const bland::ndarray &threshold_mask) {
    // Thresholded_mask holds binary information on which bins passed a threshold. Group adjacent
    // bins that passed the threshold together (connected components)
    std::vector<component> components;

    auto          thresholded_data    = threshold_mask.data_ptr<uint8_t>();
    auto          thresholded_shape   = threshold_mask.shape();
    auto          thresholded_strides = threshold_mask.strides();
    stride_helper strider(thresholded_shape, thresholded_strides);
    nd_coords     curr_coord(thresholded_shape.size(), 0);

    auto numel = threshold_mask.numel();

    std::queue<nd_coords> coord_queue;

    for (int64_t n = 0; n < numel; ++n) {
        // dereference threshold_mask w/ linear offset computed from current nd_index
        auto curr_linear = strider.to_linear_offset(curr_coord);
        if (thresholded_data[curr_linear] > 0) {
            // nd_index locations;
            coord_queue.push(curr_coord);
            component this_component;

            while (!coord_queue.empty()) {
                nd_coords idx = coord_queue.front();
                coord_queue.pop();

                // if these coordinates are valid && this belongs, keep it and add every neighbor to queue
                bool in_bounds = true;
                for (int dim = 0; dim < thresholded_shape.size(); ++dim) {
                    if (idx[dim] < 0 || idx[dim] >= thresholded_shape[dim]) {
                        in_bounds = false;
                        break;
                    }
                }
                auto coord_linear = strider.to_linear_offset(idx);
                if (in_bounds && thresholded_data[coord_linear] > 0) {
                    // This is part of the current component
                    this_component.locations.push_back(idx);
                    thresholded_data[coord_linear] = 0; // Mark as visited

                    for (int dim = 0; dim < thresholded_shape.size(); ++dim) {
                        auto next_coord = idx;
                        next_coord[dim] += 1;
                        // check if the next coordinate is valid and not visited
                        if (next_coord[dim] >= 0 && next_coord[dim] < thresholded_shape[dim] &&
                            thresholded_data[strider.to_linear_offset(next_coord)] > 0) {
                            coord_queue.push(next_coord);
                        }

                        auto prev_coord = idx;
                        prev_coord[dim] -= 1;
                        // check if the previous coordinate is valid and not visited
                        if (prev_coord[dim] >= 0 && prev_coord[dim] < thresholded_shape[dim] &&
                            thresholded_data[strider.to_linear_offset(prev_coord)] > 0) {
                            coord_queue.push(prev_coord);
                        }
                    }
                } else {
                    // Otherwise, toss it away
                    continue;
                }
            }

            components.push_back(this_component); // Assuming 's' is some statistic you compute for each component
        }
        // Increment the nd_index
        for (int dim = curr_coord.size() - 1; dim >= 0; --dim) {
            // If we're not at the end of this dim, keep going
            if (++curr_coord[dim] != thresholded_shape[dim]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                curr_coord[dim] = 0;
            }
        }
    }
    return components;
}

std::vector<component> bliss::find_components_above_threshold(doppler_spectrum &dedrifted_spectrum,
                                                              noise_stats       noise_stats,
                                                              float             snr_threshold) {
    auto hard_threshold = compute_signal_threshold(noise_stats, dedrifted_spectrum.integration_length(), snr_threshold);

    std::vector<component> components;

    // TODO: we also want to keep track of which flags contributed to *this* dedrifted spectrum so we can throw away
    // "hits" with too many (or the wrong kind) of flags
    auto &doppler_spectrum = dedrifted_spectrum.dedrifted_spectrum();
    if (doppler_spectrum.dtype() != bland::ndarray::datatype::float32) {
        throw std::runtime_error("find_components_above_threshold: dedrifted doppler spectrum was not float. Only cpu "
                                 "float is supported right now");
    }
    auto          doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto          doppler_spectrum_strides = doppler_spectrum.strides();
    stride_helper doppler_spectrum_strider(doppler_spectrum.shape(), doppler_spectrum_strides);

    auto          visited         = bland::ndarray(dedrifted_spectrum.dedrifted_spectrum().shape(),
                                  0,
                                  bland::ndarray::datatype::uint8,
                                  bland::ndarray::dev::cpu);
    auto          visited_data    = visited.data_ptr<uint8_t>();
    auto          visited_shape   = visited.shape();
    auto          visited_strides = visited.strides();
    stride_helper visited_strider(visited_shape, visited_strides);
    nd_coords     curr_coord(visited_shape.size(), 0);

    auto numel = visited.numel();

    std::queue<nd_coords> coord_queue;

    for (int64_t n = 0; n < numel; ++n) {
        // Compute linear offsets for current location to search
        auto visited_linear          = visited_strider.to_linear_offset(curr_coord);
        auto doppler_spectrum_linear = doppler_spectrum_strider.to_linear_offset(curr_coord);
        // If not visited and signal is above threshold...
        if (visited_data[visited_linear] == 0 && doppler_spectrum_data[doppler_spectrum_linear] > hard_threshold) {
            coord_queue.push(curr_coord);
            component this_component;
            this_component.max_integration = 0;

            while (!coord_queue.empty()) {
                nd_coords idx = coord_queue.front();
                coord_queue.pop();

                // if these coordinates are valid && this belongs, keep it and add every neighbor to queue
                bool in_bounds = true;
                for (int dim = 0; dim < visited_shape.size(); ++dim) {
                    if (idx[dim] < 0 || idx[dim] >= visited_shape[dim]) {
                        in_bounds = false;
                        break;
                    }
                }
                auto this_coord_visited_linear          = visited_strider.to_linear_offset(idx);
                auto this_coord_doppler_spectrum_linear = doppler_spectrum_strider.to_linear_offset(idx);
                // Test if this drift is part of the current cluster:
                // * the coordinates are valid (probably an obsolete test)
                // * we have not visited this yet (probably an obsolete test)
                // * it passes the hard threshold
                // TODO, add some more to greedily merge clusters split by noise / minor signal power drops at off
                // integrations by testing a distance metric We might even want to pass a callable if we can define this
                // well (and expose to python as callable!)
                if (in_bounds && visited_data[this_coord_visited_linear] == 0 &&
                    doppler_spectrum_data[this_coord_doppler_spectrum_linear] > hard_threshold) {
                    // This is part of the current component
                    this_component.locations.push_back(idx);
                    ++visited_data[this_coord_visited_linear]; // Mark as visited
                    // Track some stats for this cluster like the maximum value and where it is
                    if (doppler_spectrum_data[this_coord_doppler_spectrum_linear] > this_component.max_integration) {
                        this_component.max_integration = doppler_spectrum_data[this_coord_doppler_spectrum_linear];
                        this_component.index_max       = idx;
                    }

                    for (int dim = 0; dim < visited_shape.size(); ++dim) {
                        auto next_coord = idx;
                        next_coord[dim] += 1;
                        // check if the next coordinate is valid and not visited
                        if (next_coord[dim] >= 0 && next_coord[dim] < visited_shape[dim] &&
                            visited_data[visited_strider.to_linear_offset(next_coord)] == 0) {
                            coord_queue.push(next_coord);
                        }

                        auto prev_coord = idx;
                        prev_coord[dim] -= 1;
                        // check if the previous coordinate is valid and not visited
                        if (prev_coord[dim] >= 0 && prev_coord[dim] < visited_shape[dim] &&
                            visited_data[visited_strider.to_linear_offset(prev_coord)] == 0) {
                            coord_queue.push(prev_coord);
                        }
                    }
                } else {
                    // Otherwise, toss it away
                    continue;
                }
            }

            components.push_back(this_component); // Assuming 's' is some statistic you compute for each component
        }
        // Increment the nd_index
        for (int dim = curr_coord.size() - 1; dim >= 0; --dim) {
            // If we're not at the end of this dim, keep going
            if (++curr_coord[dim] != visited_shape[dim]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                curr_coord[dim] = 0;
            }
        }
    }
    return components;
}

std::vector<hit> bliss::hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, float snr_threshold) {
    std::vector<hit> hits;

    // auto threshold_mask = hard_threshold_drifts(dedrifted_spectrum.dedrifted_spectrum(),
    //                                             noise_stats,
    //                                             dedrifted_spectrum.integration_length(),
    //                                             snr_threshold);

    // // Now run connected components....
    // auto components = find_components_in_binary_mask(threshold_mask);
    auto components = find_components_above_threshold(dedrifted_spectrum, noise_stats, snr_threshold);
    fmt::print("Found {} components\n", components.size());

    for (const auto &c : components) {
        // Assume dims size 2 for now :-| (we'll get beam stuff sorted eventually)
        hit this_hit;
        this_hit.rate_index = c.index_max[0];
        this_hit.start_freq_index = c.index_max[1];

        // Start frequency in Hz is bin * Hz/bin
        this_hit.start_freq_MHz = dedrifted_spectrum.fch1() + dedrifted_spectrum.foff() * this_hit.start_freq_index;

        // TODO: sort out better names for these things as we go to whole-cadence integration
        auto drift_freq_span_bins = dedrifted_spectrum.integration_options().low_rate + this_hit.rate_index * dedrifted_spectrum.integration_options().rate_step_size;
        float drift_span_freq_Hz = drift_freq_span_bins * 1e6 * dedrifted_spectrum.foff();

        auto drift_span_time_bins = dedrifted_spectrum.integration_length();
        auto drift_span_time_sec = drift_span_time_bins * dedrifted_spectrum.tsamp();

        this_hit.drift_rate_Hz_per_sec = drift_span_freq_Hz / drift_span_time_sec;

        auto signal_power = std::pow((c.max_integration - noise_stats.noise_floor()), 2);
        auto noise_power = (noise_stats.noise_power()/std::sqrt(dedrifted_spectrum.integration_length()));
        this_hit.snr = signal_power/noise_power;
        fmt::print("s = {}, n = {} ||| so S / N = {} / {} = {} ", c.max_integration, noise_stats.noise_floor(), signal_power, noise_power, this_hit.snr);

        // At the drift rate with max SNR, find the width of this component
        // We can also integrate signal power over the entire bandwidth / noise power over bandwidth to get
        // a better picture of actual SNR rather than SNR/Hz @ peak
        this_hit.binwidth = 1;
        int64_t lower_freq_index_at_rate = this_hit.start_freq_index;
        int64_t upper_freq_index_at_rate = this_hit.start_freq_index;
        for (const auto &l : c.locations) {
            if (l[0] == this_hit.rate_index) {
                if (l[1] > upper_freq_index_at_rate) {
                    upper_freq_index_at_rate = l[1];
                } else if (l[1] < lower_freq_index_at_rate) {
                    lower_freq_index_at_rate = l[1];
                }
            }
        }
        this_hit.binwidth = upper_freq_index_at_rate - lower_freq_index_at_rate;
        this_hit.bandwidth = this_hit.binwidth * std::abs(1e6 * dedrifted_spectrum.foff());
        hits.push_back(this_hit);
    }

    return hits;
}
