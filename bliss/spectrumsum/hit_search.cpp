
#include "spectrumsum/hit_search.hpp"

// #include <algorithm>
// #include <iostream>

// using namespace bliss;

struct hit {
    int64_t lower_freq_index; /// index of the lowest (beginning) frequency of detected signal
    int64_t drift_index; /// index of the drift of detected signal
    float   integrated_power; /// integrated power of the drifting signal
};

// void bliss::hitsearch(const row_major_f32array &drift_spectrum, noise_power noise_stats, float snr)
// {
//     /********************************************************
//      ********************************************************
//      ** leaving off, I want to compute the mean / var of   **
//      ** the underlying spectrum once, then derive expected **
//      ** statistics of drift spectrum... the drift spectrum **
//      ** can then just be compared to those rather than     **
//      ** normalizing each time...                           **
//      ********************************************************
//      ********************************************************
//      */

//     std::cout << "Hit search!" << std::endl;

//     // People will likely expect hits to contain:
//     // * the lower frequency edge (convenience factor being the index of this)
//     // * the upper frequency edge (convenience factor being the index of this)
//     // * the duration of the drift (at least initially this will always be allt ime)
//     // * drift rate Hz/sec
//     // * snr estimate <- quagmire alert
//     // *
//     // TODO: think through relationship between these stats and an SNR
//     std::cout << "Drift spectrum mean is " << drift_spectrum.sum() / drift_spectrum.cols()
//               << " and working with noise stats mean of " << noise_stats.mean() << std::endl;

//     auto above_thresh = (drift_spectrum > (noise_stats.mean() + snr * noise_stats.stddev()));

//     std::vector<hit> hits_above_threshold;
//     for (int64_t drift_index = 0; drift_index < above_thresh.rows(); ++drift_index) {
//         for (int64_t freq_index = 0; freq_index < above_thresh.cols(); ++freq_index)
//             if (above_thresh(drift_index, freq_index)) {
//                 std::cout << "Got a hit: " << freq_index << " drifting " << drift_index << std::endl;
//                 hits_above_threshold.emplace_back(hit{.lower_freq_index = freq_index,
//                                                         .drift_index      = drift_index,
//                                                         .integrated_power = drift_spectrum(drift_index, freq_index)});
//             }
//     }

//     std::cout << "Got " << hits_above_threshold.size() << " hits before filtering" << std::endl;


//     // # Loop for all spectrum elements that exceed the given SNR threshold.
//     // # We offset each index value returned by np.nonzero()[0] by specstart
//     // # in order to use the returned index set on the original spectrum array.
//     // for i in (spectrum[specstart:specend] > snr_thresh).nonzero()[0] + specstart:

//     //     if logger.level == logging.DEBUG:
//     //         info_str = 'Hit found at SNR %f!\t' % (spectrum[i])
//     //         info_str += 'Spectrum index: %d, Drift rate: %f\t' % (i, drift_rate)
//     //         info_str += 'Uncorrected frequency: %f\t' % chan_freq(header, i, tdwidth, 0)
//     //         info_str += 'Corrected frequency: %f' % chan_freq(header, i, tdwidth, 1)
//     //         logger.debug(info_str)

//     //     hits += 1
//     //     if spectrum[i] > max_val.maxsnr[i]:
//     //         max_val.maxsnr[i] = spectrum[i]
//     //         max_val.maxdrift[i] = drift_rate

//     // max_val.total_n_hits[0] += hits
// }
