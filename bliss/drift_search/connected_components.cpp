
#include <drift_search/connected_components.hpp>
#include <drift_search/hit_search.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <queue>

using namespace bliss;


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
