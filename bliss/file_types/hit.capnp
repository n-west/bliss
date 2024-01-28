# To regenerate the files that are based on this schema, run:
# capnp compile -oc++ hit.capnp

# unique id generated by capnp
@0xb0d6956dc23c3717; 


# The Signal contains information about a linear signal we found.
# Some of this is redundant if the Filterbank is present, so that the Signal
# is still useful on its own.
struct Signal {
  # The frequency the hit starts at
  frequency @0 :Float64;

  # Which frequency bin the hit starts at.
  # This is relative to the coarse channel.
  index @1 :Int32;

  # How many bins the hit drifts over.
  # This counts the drift distance over the full rounded-up power-of-two time range.
  driftSteps @2 :Int32;

  # The drift rate in Hz/s
  driftRate @3 :Float64;

  # The index of drift rate in doppler spectrum
  driftIndex @10 :Int32;

  # The signal-to-noise ratio for the hit
  snr @4 :Float32;

  # Which coarse channel this hit is in
  coarseChannel @5 :Int32;

  # Which beam this hit is in. -1 for incoherent beam, or no beam
  beam @6 :Int32;

  # The number of timesteps in the associated filterbank.
  # This does *not* use rounded-up-to-a-power-of-two timesteps.
  numTimesteps @7 :Int32;

  # The total power that is normalized to calculate snr.
  # snr = (power - median) / stdev
  power @8 :Float32;

  # The total power for the same signal, calculated incoherently.
  # This is available in the stamps files, but not in the hits files.
  incoherentPower @9 :Float32;
}

# The Filterbank contains a smaller slice of the larger filterbank that we originally
# found this hit in.
struct Filterbank {
  # These fields are like the ones found in FBH5 files.
  sourceName @0 :Text;
  fch1 @1 :Float64;   # MHz
  foff @2 :Float64;   # MHz
  tstart @3 :Float64;
  tsamp @4 :Float64;
  ra @5 :Float64;     # Hours
  dec @6 :Float64;    # Degrees
  telescopeId @7 :Int32;
  numTimesteps @8 :Int32;
  numChannels @9 :Int32;

  # The length of data should be num_timesteps * num_channels.
  # Storing both of those is slightly redundant but more convenient.
  # The format is a row-major array, indexed by [timestep][channel].
  data @10 :List(Float32);

  # Additional fields that don't correspond to FBH5 headers

  # Which of the coarse channels in the file this hit is in
  coarseChannel @11 :Int32;

  # Column zero in the data corresponds to this column in the whole coarse channel
  startChannel @12 :Int32;

  # Which beam this data is from. -1 for incoherent beam, or no beam
  beam @13 :Int32;
}

# A hit without a signal indicates that we looked for a hit here and didn't find one.
# A hit without a filterbank indicates that to save space we didn't store any filterbank
# data in this file; it should be available elsewhere.
struct Hit {
  signal @0 :Signal;
  filterbank @1 :Filterbank;
}

# An event is a group of hits across a "cadence".
# It includes hits with a signal where we saw something, and hits with
# no signal where we did not see something.
struct Event {
  hits @0 :List(Hit);
}

# This is a filterbank file with detected signals (hits)
# in bliss we call this a scan with 
struct ScanDetections {
  scan @0: Filterbank;
  detections @1 :List(Signal);
}
