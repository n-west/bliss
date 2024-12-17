
BLISS Concepts
==============


BLISSdedrift is a toolkit to build configurable pipelines to find candidate technosignature `hits` and `events` in `dynamic spectra` by assuming a technosignature will be narrowband signals with nearly linear doppler drift.
`Dynamic spectra` is a common technosignature data product from radio telescopes which is similar to common spectrograms.
Starting with time-domain voltage samples, the 

BLISSdedrift is meant to build pipelines that search for narrowband doppler drifting signals.
This follows the SETI technosignature thesis that an extraterrestrial signal from an intelligent civilization would concentrate as much energy in to as narrow of a signal as possible.
The information content would be low, but successful reception requires minimal processing gain with an unmistakable direction of origin.
Since the earth rotates and moves around our sun, there will be relative motion between our radio telescopes receiving a potential signal and any transmitter not in a synchronous orbit around earth.
The goal of this package is to search for signals with very narrow bandwidth and approximately linear dopppler drifting frequency.


A linear doppler drifting narrowband signal is very similar to a linear frequency modulated (FM) chirp. A typical SETI data product is dynamic spectra, which is very similar to a spectrogram. Dynamic spectra is most often a 2d (but sometimes 3d) array of frequency content over time. Large instantaneous bandwidths (on the order of GHz) are searched with as fine of a frequency resolution as is practical to store (typically low single digit Hz). In order to manage this, telescopes will often do two stages of channelization: a coarse channelization and a fine channelization. Coarse channelization is typically performed with a polyphase filterbank and fine channelization is often a second polyphase filterbank or short-time fourier transform. The magnitude of those fine channels over time forms the dynamic spectra data product used in BLISSdedrift.


BLISSdedrift has a set of modules that map to conceptual stages of narrowband technosignature searches for doppler drifting signals as well as a hierarchy of classes that map to common radio telescope recordings. At the bottom of the hierarchy is a ``coarse_channel`` which simultaneously defines a unit of work for BLISSdedrift and is meant to represent the common procedure of radio telescopes generating dynamic spectra to do a coarse channelization early in the digital processing. A ``coarse_channel`` is a unit of a ``scan`` which contains dynamic spectra from a single astronomical source that a telescope tracked. A ``scan`` will contain at least one ``coarse_channel`` and can be part of an ``observation_target`` that is part of a ``cadence``.

The amount of data in a single ``scan``, ``observation_target``, or ``cadence`` is often more than can fit in a typical GPU's vRAM, so internally calling BLISSdedrift functions that operate on these types will often defer actual data loading or compute until the result is requested. Additionally, processing will often happen on a ``coarse_channel`` basis with little to no mixing of information between ``coarse_channel``'s. On a practical level, this means that using most of the BLISSdedrift APIs result in building a compute graph that is lazily evaluated per ``coarse_channel``.

Although ``coarse_channel``'s most often represent literal `coarse channels` which are the result of polyphase-filterbank channelizers, they are also overloaded to be the work size. Some telescopes have very large coarse channels out of their filterbanks. These are most easily processed by using a smaller work size that divides the fine channels in to equal sizes of psuedo coarse channels as `coarse_channel` objects in BLISSdedrift. The most prevalent example of this is the Parkes UWL feed which provides coarse channels consisting of 64M fine channels.



.. toctree::
   :maxdepth: 2
   :caption: Concepts:

   preprocessing
   flaggers
   noise_estimation
   dedrift
   hit_search
   event_search


Memory requirements
--------------------

To run flagging and hit search on a GPU requires the dynamic spectra, a mask, and dedrifted spectrum all exist in GPU vRAM. For
a `coarse_channel` with 2**20 (1048576) fine channels and 16 time steps that is 16777216 data points. Each dynamic spectra is a
four-byte float and has a corresponding 1 byte mask for a total of ~83MB on the GPU.

The dedrifted spectrum is sized with the same number of channels, but sized for the number of drifts. The dedrift process also tracks
flags for each channel and drift. The tracked flags are low spectral kurtosis, high spectral kurtosis, and sigma clipping which take
1 B each for a total of 3B.

The hit search process uses an another array also sized by number of drifts x number of channels to assign unique label ids for each
local maxima in dedrift spectrum. There is a small amount of 

+------------------+---------------+----------------------+---------------+-----------------+
| Step and name    | shape         | representative shape | datatype size | total size (MB) |
+==================+===============+======================+===============+=================+
| dynamic spectra  | tstep, Nchan  | 16, 2**20            | 4B            | 67 MB           |
| mask             | tstep, Nchan  | 16, 2**20            | 1B            | 16 MB           |
+------------------+---------------+----------------------+---------------+-----------------+
| dedrifted spectra| Ndrift, Nchan | 1000, 2**20          | 4B            | 4194 MB         |
| dedrifted flags  | Ndrift, Nchan | 1000, 2**20          | 3B            | 3145 MB         |
+------------------+---------------+----------------------+---------------+-----------------+
| hit search label | Ndrift, Nchan | 1000, 2**20          | 4B            | 4194 MB         |
| hit metadata     |               |                      |               |                 |
+------------------+---------------+----------------------+---------------+-----------------+
| Total            |               |                      |               | 11616 MB        |
+------------------+---------------+----------------------+---------------+-----------------+
