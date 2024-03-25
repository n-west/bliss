
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def plot_hits(cc):
    spectrum = cc.data
    spectrum = spectrum.to("cpu")
    plot_spectrum = np.log10(np.from_dlpack(spectrum) + 1e-3)
    
    plane = cc.integrated_drift_plane().integrated_drifts
    plane = plane.to("cpu")
    plot_plane = np.log10(np.from_dlpack(plane) + 1e-3)

    fch1 = cc.fch1
    foff = cc.foff
    nchans = cc.nchans
    freqs = np.linspace(fch1, fch1 + foff * nchans, nchans)

    start_time = cc.tstart * 24 * 60 * 60
    end_time = start_time + cc.tsamp * plot_spectrum.shape[0]


    # Create a gridspec instance with 10 rows
    gs = gridspec.GridSpec(20, 1)

    plt.figure()

    # Assign different rows to your subplots
    ax0 = plt.subplot(gs[:4, 0])  # Top plot gets 1 row (10% of the height)
    ax1 = plt.subplot(gs[4:12, 0], sharex=ax0)  # Middle plot gets 4 rows (40% of the height)
    ax2 = plt.subplot(gs[12:, 0], sharex=ax0)  # Bottom plot gets 5 rows (50% of the height)

    # fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(10, 10))

    spec_min = np.nanmin(plot_spectrum)
    spec_max = np.nanmax(plot_spectrum) - 1

    plane_min = np.nanmin(plot_plane) + 7
    plane_max = np.nanmax(plot_plane) - 1

    ax0.plot(freqs, plot_spectrum.sum(0))
    ax1.imshow(plot_spectrum, aspect="auto", interpolation="None", cmap="turbo", vmin=spec_min, vmax=spec_max, extent=[freqs[0], freqs[-1], end_time, start_time])
    ax2.imshow(plot_plane, aspect="auto", interpolation="None", cmap="turbo", vmin=plane_min, vmax=plane_max, extent=[freqs[0], freqs[-1], 100, -100])

    aspect_ratio = get_aspect(ax1)

    for hit in cc.hits:
        # start_time = plot_scan.tstart * 24 * 60 * 60
        # end_time = start_time + plot_scan.tsamp * plot_data.shape[0]
        start_freq = hit.start_freq_MHz
        end_freq = start_freq + (end_time - start_time) * hit.drift_rate_Hz_per_sec*1e6

        color = "red"
        # angle=-(end_time-start_time)/(1e-3 + (end_freq-start_freq)*1e6) * aspect_ratio / 45
        angle = 0
        # (end_freq-start_freq) + 
        rect = Rectangle(((start_freq), start_time), hit.bandwidth/1e6, end_time-start_time, edgecolor=color, fill=None, angle=angle, rotation_point="center")
        ax1.add_patch(rect)

    plt.tight_layout()
