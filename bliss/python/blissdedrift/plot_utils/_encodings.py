
import string # only used to get ascii_uppercase

from ..import flaggers
def get_hits_list(pipeline_object, origin_name=None):
    '''Encode hits from a bliss object type to a list of dictionaries (one dictionary per hit)
    This encoding is required for vega-based plotters

    the origin_name is generated for iterable types to form a string
    '''
    try:
        if origin_name is None:
            # Try to come up with a good one, but it probably doesn't matter
            origin_name = f"{pipeline_object.source_name}:{pipeline_object.src_dej:1.4f},{pipeline_object.src_raj:1.4f}:{pipeline_object.tstart:1.4f}"
        # This should succeed for `coarse_channel` and `scan` which can directly query `hits()`
        hits = pipeline_object.hits
        hits_list = [{
            "start_freq_MHz": hit.start_freq_MHz, "drift_rate_Hz_per_sec": hit.drift_rate_Hz_per_sec,
            "SNR": hit.snr, "power": hit.power, "bandwidth_Hz": hit.bandwidth,
            # "filter_rolloff_bins": hit.rfi_counts[flaggers.flag_values.filter_rolloff],
            # "low_sk_bins": hit.rfi_counts[flaggers.flag_values.low_spectral_kurtosis],
            # "high_sk_bins": hit.rfi_counts[flaggers.flag_values.high_spectral_kurtosis],
            "origin": origin_name,
        } for hit in hits]
        return hits_list
    except AttributeError:
        pass

    hits = []
    try:
        # This should succeed for `observation_target` which has `scans` (list of `scan`)
        for index, scan in enumerate(pipeline_object.scans):
            if origin_name is None:
                scan_origin_name = f"{pipeline_object.target_name}-{index}"
            else:
                scan_origin_name = f"{origin_name}:{pipeline_object.target_name}-{index}"
            hits.extend(get_hits_list(scan, scan_origin_name))
        return hits
    except AttributeError:
        pass

    try:
        # This should succeed for `cadence` which has `observations` (list of `obervation_target`)
        for index, obs_target in enumerate(pipeline_object.observations):
            hits.extend(get_hits_list(obs_target, f"{string.ascii_uppercase[index]}"))
        return hits
    except AttributeError:
        pass

    print("Made it to end and got nothing with hits")