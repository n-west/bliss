
#include "file_types/hits_file.hpp"

// #include "capnp/c++.capnp.h"

using namespace bliss;

void bliss::write_hits_to_file(Hit hit, std::string_view file_path) {

    // seticore does something like this:
    // auto fd = open(tmp_filename.c_str(), O_WRONLY | O_CREAT, 0664);
    // if (fd < 0) {
    //     int err = errno;
    //     fatal(fmt::format("could not open {} for writing. errno = {}", tmp_filename, err));
    // }

    // ::capnp::MallocMessageBuilder message;

    // Hit::Builder hit = message.initRoot<Hit>();

    // signal.setFrequency(hit.frequency);
    // signal.setIndex(hit.index);
    // signal.setDriftSteps(hit.drift_steps);
    // signal.setDriftRate(hit.drift_rate);
    // signal.setSnr(hit.snr);
    // signal.setCoarseChannel(hit.coarse_channel);
    // signal.setBeam(hit.beam);
    // signal.setNumTimesteps(hit.num_timesteps);
    // signal.setPower(hit.power);
    // signal.setIncoherentPower(hit.incoherent_power);

    // Filterbank::Builder filterbank = hit.getFilterbank();

    // filterbank.setSourceName(metadata.getBeamSourceName(beam));
    // filterbank.setRa(metadata.getBeamRA(beam));
    // filterbank.setDec(metadata.getBeamDec(beam));
    // filterbank.setTelescopeId(metadata.telescope_id);
    // filterbank.setFoff(metadata.foff);
    // filterbank.setTsamp(metadata.tsamp);
    // filterbank.setTstart(metadata.tstart);
    // filterbank.setNumTimesteps(metadata.num_timesteps);
    // filterbank.setCoarseChannel(coarse_channel);
    // filterbank.setBeam(beam);

    // // Find the interval [begin_index, end_index) that we actually want to copy over
    // // Pad with extra columns but don't run off the edge
    // long begin_index   = max(low_index - EXTRA_COLUMNS, 0L);
    // long end_index     = min(high_index + EXTRA_COLUMNS, metadata.coarse_channel_size) - 1;
    // long num_channels  = end_index - begin_index;
    // long coarse_offset = coarse_channel * metadata.coarse_channel_size;
    // filterbank.setNumChannels(num_channels);
    // filterbank.setFch1(metadata.fch1 + (coarse_offset + begin_index) * metadata.foff);
    // filterbank.setStartChannel(begin_index);
    // filterbank.initData(metadata.num_timesteps * num_channels);
    // auto data = filterbank.getData();

    // // Copy out the subset which would be data[:][begin_index:end_index] in numpy
    // for (long tstep = 0; tstep < metadata.num_timesteps; ++tstep) {
    //     for (long chan = 0; chan < num_channels; ++chan) {
    //         data.set(tstep * num_channels + chan, input[tstep * metadata.coarse_channel_size + begin_index + chan]);
    //     }
    // }

    // writeMessageToFd(fd, message);
}