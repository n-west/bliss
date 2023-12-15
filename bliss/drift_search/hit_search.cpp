
#include "bland/ndarray.hpp"
#include <drift_search/hit_search.hpp>

#include <fmt/core.h>
#include <cstdint>
#include <queue>

using namespace bliss;

bland::ndarray bliss::hard_threshold_drifts(const bland::ndarray &dedrifted_spectrum,
                                            const noise_stats    &noise_stats,
                                            int64_t               integration_length,
                                            float                 snr_threshold) {

    // When the signal amplitude is snr_threshold above the noise floor, we have a 'prehit' (a signal that naively
    // passes a hard threshold) that is when S/N > snr_threshold Given a noise floor estimate of nf, signal amplitude s,
    // noise amplitude n... S = (s - nf)**2 N = (n)**2         our estimate has already taken in to account noise floor
    // (s-nf)/(n) > sqrt(snr_threshold)
    // s-nf > n * sqrt(snr_threshold)
    // s > nf + sqrt(N * snr_threshold)
    // Since the noise power was estimate before integration, it also decreases by sqrt of integration length
    auto integration_adjusted_noise_power = noise_stats.noise_power() / std::sqrt(integration_length);
    auto hard_threshold = noise_stats.noise_floor() + std::sqrt(integration_adjusted_noise_power * snr_threshold);

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


// std::vector<component> bliss::find_binary_components(const bland::ndarray &threshold_mask) {
// }

// std::vector<component> bliss::find_binary_components_above(const bland::ndarray &threshold_mask, float threshold) {
// }

std::vector<component> bliss::find_components(const bland::ndarray &threshold_mask) {
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
        if (thresholded_data[curr_linear] > 0) { // if there's a connection here, go in
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
                    thresholded_data[coord_linear] = 0; // Mark as visited



                    for (int dim = 0; dim < thresholded_shape.size(); ++dim) {
                        // create a copy of the current coordinate
                        auto next_coord = idx;
                        // increment the dimension by one
                        next_coord[dim] += 1;
                        // check if the next coordinate is valid and not visited
                        if (next_coord[dim] >= 0 && next_coord[dim] < thresholded_shape[dim] && thresholded_data[strider.to_linear_offset(next_coord)] > 0) {
                            // push it to the queue
                            coord_queue.push(next_coord);
                        }
                        // create another copy of the current coordinate
                        auto prev_coord = idx;
                        // decrement the dimension by one
                        prev_coord[dim] -= 1;
                        // check if the previous coordinate is valid and not visited
                        if (prev_coord[dim] >= 0 && prev_coord[dim] < thresholded_shape[dim] && thresholded_data[strider.to_linear_offset(prev_coord)] > 0) {
                            // push it to the queue
                            coord_queue.push(prev_coord);
                        }
                    }
                    this_component.locations.push_back(idx);
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

std::vector<hit> bliss::hit_search(doppler_spectrum dedrifted_spectrum, noise_stats noise_stats, float snr_threshold) {
    std::vector<hit> hits;

    auto threshold_mask = hard_threshold_drifts(dedrifted_spectrum.dedrifted_spectrum(),
                                                noise_stats,
                                                dedrifted_spectrum.integration_length(),
                                                snr_threshold);

    // Now run connected components....
    auto components = find_components(threshold_mask);

    fmt::print("Found {} components\n", components.size());

    return hits;
}
