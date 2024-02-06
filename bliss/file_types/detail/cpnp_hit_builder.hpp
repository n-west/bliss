#pragma once

#include <core/hit.hpp>
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

} // namespace bliss::detail
