#pragma once

#if BLISS_USE_CAPNP

#include "hit.capnp.h"
#include <string_view>

namespace bliss {

// TODO: this is all filler until we really grok how we should be usign capnproto
void write_hits_to_file(Hit hit, std::string_view file_path);

} // namespace bliss

#endif // BLISS_USE_CAPNP
