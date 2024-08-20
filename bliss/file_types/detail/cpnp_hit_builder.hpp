#pragma once

#include <core/hit.hpp>
#include <core/coarse_channel.hpp>
#include <core/scan.hpp>
#include <core/cadence.hpp>
#include "hit.capnp.h"

#include <stdexcept>

#include <capnp/c++.capnp.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/exception.h>


namespace bliss::detail {

/**
 * A single source for converting our native `bliss::hit` type to a cap'n proto `signal` through a capnp builder
 */
void bliss_hit_to_capnp_signal_message(Signal::Builder &signal_builder, hit this_hit);

hit capnp_signal_message_to_bliss_hit(const Signal::Reader &signal_reader);

void bliss_coarse_channel_to_capnp_coarse_channel_message(CoarseChannel::Builder &cc_builder, coarse_channel);

coarse_channel capnp_coarse_channel_message_to_bliss_coarse_channel(CoarseChannel::Reader &cc_reader);

void bliss_scan_to_capnp_scan(Scan::Builder &cc_builder, scan);

scan capnp_scan_message_to_bliss_scan(Scan::Reader &cc_reader);

void bliss_observation_target_to_capnp_observation_target_message(ObservationTarget::Builder &obstarget_builder, observation_target);

observation_target capnp_observation_target_message_to_bliss_observation_target(ObservationTarget::Reader &obstarget_reader);

void bliss_cadence_to_capnp_cadence_message(Cadence::Builder &obstarget_builder, cadence);

cadence capnp_cadence_message_to_cadence(Cadence::Reader &obstarget_reader);

} // namespace bliss::detail
