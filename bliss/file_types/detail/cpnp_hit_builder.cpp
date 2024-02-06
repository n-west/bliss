
#include "cpnp_hit_builder.hpp"
#include <core/hit.hpp>

using namespace bliss;
using namespace bliss::detail;

void bliss::detail::bliss_hit_to_capnp_signal_message(Signal::Builder &signal_builder, hit this_hit) {
    signal_builder.setFrequency(this_hit.start_freq_MHz);
    signal_builder.setIndex(this_hit.start_freq_index);
    signal_builder.setDriftSteps(
            this_hit.time_span_steps); // redundant with numTimesteps but was a rounded-up power of 2 in seticore
    signal_builder.setDriftRate(this_hit.drift_rate_Hz_per_sec);
    signal_builder.setDriftIndex(this_hit.rate_index);
    // this_hit.rate_index;
    // signal_builder.setIndex
    signal_builder.setSnr(this_hit.snr);
    // signal_builder.setCoarseChannel(); // not yet looking at multiple coarse channels
    // signal_builder.setBeam(); // not yet looking at multiple beams
    signal_builder.setNumTimesteps(this_hit.time_span_steps); // for us, redundant with driftSteps
    // the power and incoherentpower are redundant in bliss, but were not in seticore
    // the seticore power was normalized for computing an SNR, the incoherentpower was just the integrated power
    signal_builder.setPower(this_hit.power);
    signal_builder.setIncoherentPower(this_hit.power);
    signal_builder.setStartTime(this_hit.start_time_sec);
    signal_builder.setDurationSeconds(this_hit.duration_sec);

    // These currently aren't in the capn proto definition to be serialized
    // double  bandwidth;
    // int64_t binwidth;
    // rfi     rfi_counts;
}

hit bliss::detail::capnp_signal_message_to_bliss_hit(const Signal::Reader &signal_reader) {
    hit this_hit;
    this_hit.start_freq_index      = signal_reader.getIndex();
    this_hit.start_freq_MHz        = signal_reader.getFrequency();
    this_hit.rate_index            = signal_reader.getDriftIndex();
    this_hit.drift_rate_Hz_per_sec = signal_reader.getDriftRate();
    this_hit.power                 = signal_reader.getPower();
    // this_hit.time_span_steps = signal_reader.getDriftSteps(); // in the cap'n proto definition, this is a
    // rounded up power-of-2 value
    this_hit.time_span_steps = signal_reader.getNumTimesteps();
    this_hit.snr             = signal_reader.getSnr();

    this_hit.start_time_sec = signal_reader.getStartTime();
    this_hit.duration_sec = signal_reader.getDurationSeconds();
    // The following are not (currently) in the capn proto definition
    // this_hit.bandwidth;
    // this_hit.binwidth;
    // this_hit.rfi_counts;

    return this_hit;
}
