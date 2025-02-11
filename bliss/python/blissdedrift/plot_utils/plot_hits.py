
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

def plot_hits(cc, focus_hits=None, all_hits=None, frequency_padding_Hz=500, snr=None):
    '''
    Return a list of plots centered on each `focus_hit` with all_hits visible when within
    the frequency bounds of the given plot. When `focus_hits` is None, one plot of the coarse
    channel is return with all_hits shown.

    plot_hits(cc) will return a single plot of the coarse channel with all hits shown.
    plot_hits(cc, all_hits=cc.hits) is equivalent to calling plot_hits(cc)
    '''
    spectrum = cc.data
    spectrum = spectrum.to("cpu")
    plot_spectrum = np.log10(np.from_dlpack(spectrum) + 1e-3)
    
    idp = cc.integrated_drift_plane()
    plane = idp.integrated_drifts
    sorted_drifts = sorted(idp.drift_rate_info(), key=lambda d: d.drift_rate_Hz_per_sec)
    min_drift = sorted_drifts[0].drift_rate_Hz_per_sec
    max_drift = sorted_drifts[-1].drift_rate_Hz_per_sec

    frequency_padding_MHz = frequency_padding_Hz * 1e-6
    
    plane = plane.to("cpu")
    plot_plane = np.log10(np.from_dlpack(plane) + 1e-3)

    fch1 = cc.fch1
    foff = cc.foff
    nchans = cc.nchans
    tsamp = cc.tsamp
    freqs = np.linspace(fch1, fch1 + foff * nchans, nchans)

    start_time = cc.tstart * 24 * 60 * 60
    end_time = start_time + tsamp * plot_spectrum.shape[0]
    end_time -= start_time
    start_time -= start_time

    if focus_hits is None and all_hits is None:
        all_hits = cc.hits


    if focus_hits is None:
        # one macro plot of all hits
        gs = gridspec.GridSpec(20, 1)
        fig = plt.figure()
        # Assign different rows to your subplots
        ax0 = fig.add_subplot(gs[:4, 0])  # Top plot gets 1 row (10% of the height)
        ax1 = fig.add_subplot(gs[4:12, 0], sharex=ax0)  # Middle plot gets 4 rows (40% of the height)
        ax2 = fig.add_subplot(gs[12:, 0], sharex=ax0)  # Bottom plot gets 5 rows (50% of the height)

        spec_min = np.nanmin(plot_spectrum)
        spec_max = np.nanmax(plot_spectrum)

        plane_min = np.nanmin(plot_plane)
        plane_max = np.nanmax(plot_plane)

        ax0.plot(freqs, plot_spectrum.sum(0))
        ax1.imshow(plot_spectrum, aspect="auto", interpolation="None", cmap="turbo", vmin=spec_min, vmax=spec_max, extent=[freqs[0], freqs[-1], end_time, start_time])
        ax2.imshow(plot_plane, aspect="auto", interpolation="None", cmap="turbo", vmin=plane_min, vmax=plane_max, extent=[freqs[0], freqs[-1], max_drift, min_drift])

        for hit in all_hits:
            start_freq = hit.start_freq_MHz
            end_freq = start_freq + (end_time - start_time) * hit.drift_rate_Hz_per_sec*1e-6

            lower_edge = min(start_freq, end_freq)
            upper_edge = max(start_freq, end_freq)
            if foff < 0:
                lower_edge -= hit.bandwidth*1e-6/2
                upper_edge += hit.bandwidth*1e-6/2
            else:
                lower_edge += hit.bandwidth*1e-6/2
                upper_edge -= hit.bandwidth*1e-6/2

            color = "white"
            angle = 0
            # rect = Rectangle((lower_edge, start_time), upper_edge-lower_edge, (end_time-start_time)-tsamp*.05, edgecolor=color, fill=None, angle=angle, rotation_point="center")
            # ax1.add_patch(rect)
            # ax1.plot([start_freq, end_freq], [start_time, end_time], color="red", lw=1, path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])

            # The focused hit (this is probably in all_hits and we want the red line to come on top)
            rect = Rectangle((lower_edge, start_time + tsamp*.05), upper_edge-lower_edge, (end_time-start_time)-tsamp*.05, edgecolor=color, fill=None, angle=angle, rotation_point="center")
            ax1.add_patch(rect)
            ax1.plot([start_freq, end_freq], [start_time, end_time], color="red", lw=1, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

        return [fig,]

    # From here on we are generating a plot per hit in focus_hits
    all_figs = []
    for hit in focus_hits:
        start_freq = hit.start_freq_MHz
        end_freq = start_freq + (end_time - start_time) * hit.drift_rate_Hz_per_sec*1e-6

        # Extend freq bounds by the bandwidth + 500 Hz, keeping the hit centered
        min_freq = min(start_freq, end_freq) - hit.bandwidth*1e-6/2 - frequency_padding_MHz/2
        max_freq = max(start_freq, end_freq) + hit.bandwidth*1e-6/2 + frequency_padding_MHz/2
        if max_freq - min_freq < frequency_padding_MHz:
            # Ensure plots show at minimum of frequency padding
            difference = (max_freq - min_freq) - frequency_padding_MHz
            max_freq += difference/2
            min_freq -= difference/2

        plot_start_freq_index = abs(min_freq - fch1)/abs(foff)
        plot_end_freq_index = abs(max_freq - fch1)/abs(foff)
        plot_lower_freq_index = int(min(plot_start_freq_index, plot_end_freq_index))
        plot_upper_freq_index = int(max(plot_start_freq_index, plot_end_freq_index))

        # Create a gridspec instance with 10 rows
        gs = gridspec.GridSpec(20, 1)

        fig = plt.figure()

        fig.suptitle(f"Hit starting at {start_freq:1.6f} MHz with drift rate {hit.drift_rate_Hz_per_sec:1.2f} Hz/sec")
        # Assign different rows to your subplots
        ax0 = fig.add_subplot(gs[:4, 0])  # Top plot gets 1 row (20% of the height)
        ax1 = fig.add_subplot(gs[4:12, 0], sharex=ax0)  # Middle plot gets 4 rows (40% of the height)
        ax2 = fig.add_subplot(gs[12:, 0], sharex=ax0)  # Bottom plot gets 5 rows (40% of the height)

        hit_freqs = freqs[plot_lower_freq_index:plot_upper_freq_index]
        hit_spectrum = plot_spectrum[:, plot_lower_freq_index:plot_upper_freq_index]
        hit_plane = plot_plane[:, plot_lower_freq_index:plot_upper_freq_index]
        
        spec_min = np.nanmin(hit_spectrum)
        spec_max = np.nanmax(hit_spectrum)

        plane_min = np.nanmin(hit_plane)
        plane_max = np.nanmax(hit_plane)

        ax0.plot(hit_freqs, hit_spectrum.sum(0))
        ax1.imshow(hit_spectrum, aspect="auto", interpolation="None", cmap="cividis", vmin=spec_min, vmax=spec_max, extent=[hit_freqs[0], hit_freqs[-1], end_time, start_time])
        if snr is not None:
            # hit_plane is already in log10 scale
            dedrift_plane = hit_plane > np.log10(snr)
        else:
            dedrift_plane = hit_plane
        ax2.imshow(dedrift_plane, aspect="auto", interpolation="None", cmap="cividis", vmin=plane_min, vmax=plane_max, extent=[hit_freqs[0], hit_freqs[-1], max_drift, min_drift])

        center_freq = (min_freq + max_freq) / 2
        ticks_freq = np.linspace(min_freq, max_freq, 5)
        if (abs(max_freq - min_freq)) < .002:
            tick_labels = [f"{(f - center_freq) * 1e6:.0f}" for f in ticks_freq]
            units = "Hz"
        else:
            tick_labels = [f"{(f - center_freq) * 1e3:.0f}" for f in ticks_freq]
            units = "kHz"
        ax0.set_xticks(ticks_freq)
        ax0.set_xticklabels(tick_labels)
        ax2.set_xlabel(f"Offset from {center_freq:.6f} MHz ({units})")
        # ax0.set_xlim(min_freq, max_freq)

        ax1.set_ylabel("Time (s)")
        ax2.set_ylabel("Drift Rate (Hz/sec)")

        color = "red"
        angle = 0
        # start_freq and the box width should be adjusted by the bandwidth
        lower_edge = min(start_freq, end_freq)
        upper_edge = max(start_freq, end_freq)
        if foff < 0:
            lower_edge -= hit.bandwidth*1e-6/2
            upper_edge += hit.bandwidth*1e-6/2
        else:
            lower_edge += hit.bandwidth*1e-6/2
            upper_edge -= hit.bandwidth*1e-6/2

        if all_hits is None:
            all_hits = focus_hits

        for bg_hit in all_hits:
            bg_start_freq = bg_hit.start_freq_MHz
            bg_end_freq = bg_start_freq + (end_time - start_time) * bg_hit.drift_rate_Hz_per_sec*1e-6

            # If the bg_hit is within the freq ranges of this plot, show it
            if bg_start_freq > min_freq and bg_start_freq < max_freq or bg_end_freq > min_freq and bg_end_freq < max_freq:
                ax1.plot([bg_start_freq, bg_end_freq], [start_time, end_time],  color="none", path_effects=[pe.Stroke(linewidth=.25, foreground='black'), pe.Stroke(linewidth=.125, foreground='red'), pe.Normal()])
                bg_lower_edge = min(bg_start_freq, bg_end_freq)
                bg_upper_edge = max(bg_start_freq, bg_end_freq)
                if foff < 0:
                    bg_lower_edge -= bg_hit.bandwidth*1e-6/2
                    bg_upper_edge += bg_hit.bandwidth*1e-6/2
                else:
                    bg_lower_edge += bg_hit.bandwidth*1e-6/2
                    bg_upper_edge -= bg_hit.bandwidth*1e-6/2

                rect = Rectangle((bg_lower_edge, start_time + tsamp*.05), bg_upper_edge-bg_lower_edge, (end_time-start_time)-tsamp*.05, edgecolor="black", fill=None, angle=angle, rotation_point="center")
                ax1.add_patch(rect)

        # The focused hit (this is probably in all_hits and we want the red line to come on top)
        rect = Rectangle((lower_edge, start_time + tsamp*.05), upper_edge-lower_edge, (end_time-start_time)-tsamp*.05, edgecolor=color, fill=None, angle=angle, rotation_point="center")
        ax1.add_patch(rect)
        ax1.plot([start_freq, end_freq], [start_time, end_time], color="red", lw=1, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

        fig.set_tight_layout(True)
        all_figs.append(fig)

    return all_figs
