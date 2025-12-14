import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from .wise import get_wise_lc_data

# data directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# style for ZTF filters
colors = {"ZTF_g":"green","ZTF_r":"red","ZTF_i":"orange"}
markers = {"ZTF_g":"o","ZTF_r":"X","ZTF_i":"D"}

# params df
params = pd.read_csv(DATA_DIR / "TableA_full_machine_readable_params.csv")

def plot_ztf_lc(SN_det, SN_nondet, oid, xlim=(None, None), ax=None, show=True, flux=False):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Loop over whatever filters are actually present
    for fid in sorted(SN_det.fid.dropna().unique()):
        color = colors.get(fid, "black")
        marker = markers.get(fid, "o")

        # --- Detections ---
        mask_det = (SN_det.fid == fid) & SN_det.magpsf.notna()
        if mask_det.any():
            ax.errorbar(
                SN_det.loc[mask_det, "mjd"],
                SN_det.loc[mask_det, "magpsf"],
                yerr=SN_det.loc[mask_det, "sigmapsf"],
                c=color, label=fid,
                marker=marker, linestyle='none'
            )

        # --- Non-detections (limits) ---
        mask_nondet = (SN_nondet.fid == fid) & (SN_nondet.diffmaglim > 0)
        if mask_nondet.any():
            ax.scatter(
                SN_nondet.loc[mask_nondet, "mjd"],
                SN_nondet.loc[mask_nondet, "diffmaglim"],
                c=color, alpha=0.5, marker='v',
                label=f"lim.mag. {label}"
            )

    ax.set_title(oid, fontsize=16)
    ax.set_xlabel("MJD", fontsize=14)
    ax.set_ylabel("Apparent magnitude", fontsize=14)

    # Flip y-axis so brighter = up
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_ztf_forced_lc(res, oid, xlim=(None, None), ax=None, show=True, flux=False,
                   ylim=(None, None), SNU=5.0):
    """
    Plot ZTF forced photometry light curves (apparent magnitude or flux).
    Note: Magnitude plots invert the y-axis (bright = up).
    """

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))
        created_ax = True

    all_mjd = []
    all_y = []

    # avoid duplicate legend entries for limits per filter
    shown_limit_label = set()

    # if flux=True, check if mJy fields are available
    use_mJy = flux and any(("flux_mJy" in d) for d in res.values())

    for filt, data in res.items():
        color = colors.get(filt,"black")

        # Detections
        mask_det = ~np.isnan(data["mag"])
        if np.any(mask_det):
            if flux:
                if "flux_mJy" in data and "flux_err_mJy" in data:
                    yvals = data["flux_mJy"][mask_det]
                    yerrs = data["flux_err_mJy"][mask_det]
                else:
                    yvals = data["flux"][mask_det]       # DN fallback
                    yerrs = data["flux_err"][mask_det]   # DN fallback
            else:
                yvals = data["mag"][mask_det]
                yerrs = data["mag_err"][mask_det]

            ax.errorbar(
                data["mjd"][mask_det],
                yvals,
                yerr=yerrs,
                color=color, label=filt,
                marker = markers.get(filt, "o"),
                linestyle='none'
            )
            all_mjd.extend(data["mjd"][mask_det])
            all_y.extend(yvals)

        # Non-detections (upper limits)
        mask_nondet = np.isnan(data["mag"])

        if np.any(mask_nondet):
            if flux:
                if use_mJy and "lim_flux_mJy" in data:
                    yvals_lim = data["lim_flux_mJy"][mask_nondet]
                else:
                    yvals_lim = SNU * data['flux_err'][mask_nondet]
            else:
                if "mag_ul" in data:
                    yvals_lim = data["mag_ul"][mask_nondet]
                else:
                    yvals_lim = data.get("limiting_mag", np.full_like(data["mjd"], np.nan))[mask_nondet]

            # Only plot finite limits
            finite = np.isfinite(yvals_lim)

            ax.scatter(
                data["mjd"][mask_nondet][finite],
                yvals_lim[finite],
                marker="v", alpha=0.5, color=color,
                label=f"lim. mag {filt}" if filt not in shown_limit_label else None,
                s=40
            )
            shown_limit_label.add(filt)
            all_mjd.extend(data["mjd"][mask_nondet][finite])
            all_y.extend(yvals_lim[finite])

    # Limit axis ranges if specified
    if all_mjd and all_y:
        min_mjd = xlim[0] if xlim[0] is not None else None
        max_mjd = xlim[1] if xlim[1] is not None else None

        if min_mjd is not None and max_mjd is not None:
            ax.set_xlim(min_mjd, max_mjd)

        if ylim[0] is not None and ylim[1] is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            # y-range: min/max mags + padding
            ymin = np.nanmin(all_y) - 2.0
            ymax = np.nanmax(all_y) + 1.0
            if flux:
                ax.set_ylim(ymin, ymax)
            else:   
                ax.set_ylim(ymax, ymin)  # flip so bright = up

    if created_ax:
        ax.set_title(f"ZTF Light Curve: {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        if flux:
            ax.set_ylabel("Flux (mJy)", fontsize=14)
        else:
            ax.set_ylabel("Apparent magnitude", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.4)
        
        if show:
            plt.show()

    if not created_ax:
        return ax
    
def plot_wise_lc(res, oid, xlim=(None, None), ax=None, show=True, show_baselines=False):
    """
    Plot WISE/NEOWISE light curves, optionally subtracting parity baselines.
    """

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))
        created_ax = True

    colors = {1:"navy",2:"dodgerblue"}
    markers = {1:"s",2:"s"}

    for fid, data in zip([1,2], [("b1_times","b1_fluxes","b1_fluxerrs"), ("b2_times","b2_fluxes","b2_fluxerrs")]):
        color = colors.get(fid,"black")

        # --- Detections ---
        mask_det = ~np.isnan(res[data[1]])
        if np.any(mask_det):
            ax.errorbar(
                res[data[0]][mask_det],
                res[data[1]][mask_det],
                yerr=res[data[2]][mask_det],
                fmt="o", color=color, label=f"W{fid}",
                marker = markers.get(fid, "o"),
            )
            even_base = res[f"{data[0][:2]}_even_baseline"]
            odd_base = res[f"{data[0][:2]}_odd_baseline"]

            # optional show baselines on plot
            if show_baselines:
                ax.axhline(even_base, color=color, linestyle='--', alpha=0.5, label=f"{fid} even baseline")
                ax.axhline(odd_base, color=color, linestyle=':', alpha=0.5, label=f"{fid} odd baseline")

    if created_ax:
        ax.set_title(f"WISE Light Curve: {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        ax.set_ylabel("WISE flux (mJy)", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.4)

        if xlim[0] is not None and xlim[1] is not None:
            ax.set_xlim(xlim[0], xlim[1])
        
        if show:
            plt.show()

    if not created_ax:
        return ax
    
def plot_combined_lc(ztf_res, wise_res, oid, xlim=(None, None), ztf_flux=True, mode="stacked", scale_wise=True,
                     baseline_ref="ztf", baseline_dt=100, ref_band="r", logy=True, savepath=None, mark_tail_start=False,
                     mark_plateau_end=False):
    """
    Plot ZTF + WISE light curves.
    mode: "stacked" or "overlay"
    """

    # Helper: log-safe clipping
    def _log_safe(y, yerr=None):
        y = np.asarray(y, dtype=float)
        if np.all(~np.isfinite(y)) or np.all(y <= 0):
            return y, yerr
        positive = y[y > 0]
        floor = np.nanmin(positive) * 0.1
        y_clipped = np.where(y > 0, y, floor)
        if yerr is not None:
            # prevent errorbars from crossing below floor
            yerr = np.minimum(yerr, y_clipped * 0.95)
            return y_clipped, yerr
        return y_clipped, None
    

    # x-axis range, use from Alerce detections if available
    if xlim == (None, None):
        xlim=(ztf_res["lc_det"].mjd.min(), ztf_res["lc_det"].mjd.max())
        min_mjd = xlim[0] - 30 if xlim[0] is not None else None
        max_mjd = xlim[1] + 100 if xlim[1] is not None else None
    else:
        min_mjd, max_mjd = xlim

    if mode == "stacked":
        # Create two-panel figure
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10,8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # Top: ZTF
        if ztf_flux:
            if logy:
                for band, data in ztf_res["forced"].items():
                    data["flux_mJy"], data["flux_err_mJy"] = _log_safe(
                        data["flux_mJy"], data["flux_err_mJy"]
                    )
            plot_ztf_forced_lc(ztf_res['forced'], oid=ztf_res['oid'], xlim=(min_mjd, max_mjd), ax=ax1, show=False, flux=True)
            ax1.set_ylabel("ZTF flux (mJy)", fontsize=14)
        else:
            plot_ztf_forced_lc(ztf_res['forced'], oid=ztf_res['oid'], xlim=(min_mjd, max_mjd), ax=ax1, show=False)
            ax1.set_ylabel("Apparent magnitude", fontsize=14)

        if logy and ztf_flux:
            ax1.set_yscale("log")

        ax1.grid(True, alpha=0.4)

        # Bottom: WISE
        if wise_res == {}:
                pass
        else:
            if logy:
                for band in [1, 2]:
                    flux_key = f"b{band}_fluxes"
                    fluxerr_key = f"b{band}_fluxerrs"
                    wise_res[flux_key], wise_res[fluxerr_key] = _log_safe(
                        wise_res[flux_key], wise_res[fluxerr_key]
                    )

            plot_wise_lc(wise_res, oid=ztf_res['oid'], xlim=(min_mjd, max_mjd), ax=ax2, show=False)
            ax2.set_ylabel("WISE flux (mJy)", fontsize=14)
            ax2.grid(True, alpha=0.4)
            ax2.set_xlabel("MJD", fontsize=14)

            if logy:
                ax2.set_yscale("log")
        
        # Change style for stacked
        # 1. Fix scatter points (PathCollection)
        for ax in [ax1, ax2]:
            for col in ax.collections:
                if isinstance(col, mcoll.PathCollection):
                    fcs = col.get_facecolors()
                    if fcs is None or len(fcs) == 0:
                        continue
                    r, g, b, _ = fcs[0]
                    col.set_facecolor((r, g, b, 0.3))
                    col.set_edgecolor((r, g, b, 0.3))
                    col.set_linewidth(1.2)
                    col.set_sizes([20])

            # 2. Fix errorbar markers (Line2D)
            for line in ax.lines:
                mfc = line.get_markerfacecolor()
                mec = line.get_markeredgecolor()
                if mfc is None or mfc == "none":
                    continue
                if isinstance(mfc, (tuple, list)) and len(mfc) == 4:
                    r, g, b, _ = mfc
                elif isinstance(mfc, str):
                    import matplotlib.colors as mcolors
                    r, g, b, _ = mcolors.to_rgba(mfc)
                else:
                    continue
                line.set_markerfacecolor((r, g, b, 0.3))   # semi-transparent fill
                line.set_markeredgecolor((r, g, b, 1.0))   # solid outline
                line.set_markeredgewidth(1.2)
                line.set_markersize(7)

        ax1.legend()
        ax2.legend()
        fig.subplots_adjust(hspace=0.0)
        fig.suptitle(f"ZTF + WISE Light Curve: {oid}", fontsize=16, y=0.93)

        if mark_tail_start:
            m = params[['name', 'plateauend', 'tailstart']].dropna()
            m_dict = dict(zip(m['name'].astype(str), m['tailstart'].astype(float)))
            if oid in m_dict:
                tail_start = m_dict[oid]
                ax1.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax2.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax1.text(
                    tail_start-2,          # x position (data coords)
                    0.95,                 # y as a fraction of the axes height
                    f"Tail Start ({tail_start:.1f})",        # text
                    rotation=90,          # vertical text
                    va="top",             # align text relative to its position
                    ha="center",
                    fontsize=9,
                    color="black",
                    transform=ax1.get_xaxis_transform(),  # x in data, y in axes coords
                    clip_on=False,
                )

        if mark_plateau_end:
            m = params[['name', 'plateauend', 'tailstart']].dropna()
            m_dict = dict(zip(m['name'].astype(str), m['plateauend'].astype(float)))
            if oid in m_dict:
                tail_start = m_dict[oid]
                ax1.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax2.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax1.text(
                    tail_start-2,          # x position (data coords)
                    0.95,                 # y as a fraction of the axes height
                    f"Plateau End ({tail_start:.1f})",        # text
                    rotation=90,          # vertical text
                    va="top",             # align text relative to its position
                    ha="center",
                    fontsize=9,
                    color="black",
                    transform=ax1.get_xaxis_transform(),  # x in data, y in axes coords
                    clip_on=False,
                )

        if savepath:
            plt.savefig(savepath, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {savepath}")
        else:
            plt.show()

    elif mode == "overlay":
        # Single panel figure
        fig, ax = plt.subplots(figsize=(10,6))

        if ztf_flux:
            if logy:
                for band, data in ztf_res["forced"].items():
                    data["flux_mJy"], data["flux_err_mJy"] = _log_safe(
                        data["flux_mJy"], data["flux_err_mJy"]
                    )

            plot_ztf_forced_lc(ztf_res['forced'], oid=ztf_res['oid'], xlim=(min_mjd, max_mjd), ax=ax, show=False, flux=True)
        else:
            plot_ztf_forced_lc(ztf_res['forced'], oid=ztf_res['oid'], xlim=(min_mjd, max_mjd), ax=ax, show=False)

        if wise_res == {}:
            pass
        else:
            if logy:
                for band in [1, 2]:
                    flux_key = f"b{band}_fluxes"
                    fluxerr_key = f"b{band}_fluxerrs"
                    wise_res[flux_key], wise_res[fluxerr_key] = _log_safe(
                        wise_res[flux_key], wise_res[fluxerr_key]
                    )
            plot_wise_lc(wise_res, oid=ztf_res['oid'], xlim=(min_mjd, max_mjd), ax=ax, show=False)

        # Change style for overlay
        # 1. Fix scatter points (PathCollection)
        for col in ax.collections:
            if isinstance(col, mcoll.PathCollection):
                fcs = col.get_facecolors()
                if fcs is None or len(fcs) == 0:
                    continue
                r, g, b, _ = fcs[0]
                col.set_facecolor((r, g, b, 0.3))
                col.set_edgecolor((r, g, b, 0.3))
                col.set_linewidth(1.2)
                col.set_sizes([20])

        # 2. Fix errorbar markers (Line2D)
        for line in ax.lines:
            mfc = line.get_markerfacecolor()
            mec = line.get_markeredgecolor()
            if mfc is None or mfc == "none":
                continue
            if isinstance(mfc, (tuple, list)) and len(mfc) == 4:
                r, g, b, _ = mfc
            elif isinstance(mfc, str):
                import matplotlib.colors as mcolors
                r, g, b, _ = mcolors.to_rgba(mfc)
            else:
                continue
            line.set_markerfacecolor((r, g, b, 0.3))   # semi-transparent fill
            line.set_markeredgecolor((r, g, b, 1.0))   # solid outline
            line.set_markeredgewidth(1.2)
            line.set_markersize(7)

        ax.set_title(f"ZTF + WISE Light Curve: {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        ax.set_ylabel("Flux (mJy)", fontsize=14)
        
        handles, labels = ax.get_legend_handles_labels()
        # keep only ZTF detections and WISE bands
        keep = ["ZTF_g", "ZTF_r", "ZTF_i", "W1", "W2"]
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep]

        if filtered:
            handles, labels = zip(*filtered)
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))

        ax.grid(True, alpha=0.4)

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0, max(ax.get_ylim()))

        if mark_tail_start:
            m = params[['name', 'plateauend', 'tailstart']].dropna()
            m_dict = dict(zip(m['name'].astype(str), m['tailstart'].astype(float)))
            if oid in m_dict:
                tail_start = m_dict[oid]
                ax.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax.text(
                    tail_start-2,          # x position (data coords)
                    0.95,                 # y as a fraction of the axes height
                    "Tail Start",        # text
                    rotation=90,          # vertical text
                    va="top",             # align text relative to its position
                    ha="center",
                    fontsize=9,
                    color="black",
                    transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
                    clip_on=False,
                )
        if mark_plateau_end:
            m = params[['name', 'plateauend', 'tailstart']].dropna()
            m_dict = dict(zip(m['name'].astype(str), m['plateauend'].astype(float)))
            if oid in m_dict:
                tail_start = m_dict[oid]
                ax.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                ax.text(
                    tail_start-2,          # x position (data coords)
                    0.95,                 # y as a fraction of the axes height
                    f"Plateau End ({tail_start:.1f})",        # text
                    rotation=90,          # vertical text
                    va="top",             # align text relative to its position
                    ha="center",
                    fontsize=9,
                    color="black",
                    transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
                    clip_on=False,
                )

        if savepath:
            plt.savefig(savepath, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {savepath}")
        else:
            plt.show()

    else:
        raise ValueError("Invalid mode. Choose 'stacked' or 'overlay'.")


def plot_tail_models(oid, pred_df, pred_curves, snr_min=3.0, ax=None, show_ls=True, show_hbm=True,
                plot_wise=True, xlim=(None, None), ylim=(None, None)):
    """
    Plot light curve with least-squares and HBM model fits overlaid.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))

    sn_df = df[df['name'] == oid].copy()
    bands = sn_df['band'].unique()

    ztf_res = pd.read_pickle(DATA_DIR / "ztf_forced_photometry" / f"{oid}_forced.pkl")
    markers = {"ZTF_g": "o", "ZTF_r": "X", "ZTF_i": "D"}
    colors = {"ZTF_g": "tab:green", "ZTF_r": "tab:red", "ZTF_i": "tab:orange"}

    t0 = sn_df['t0'].iloc[0]

    for band in bands:
        band_mask = sn_df['band'] == band

        phase = sn_df.loc[band_mask, "phase"].values
        flux = sn_df.loc[band_mask, "flux_mJy"].values
        flux_err = sn_df.loc[band_mask, "flux_err_mJy"].values

        snr = flux / flux_err

        mask = (
            np.isfinite(phase)
            & np.isfinite(flux)
            & np.isfinite(flux_err)
            & (flux > 0)
            & (flux_err > 0)
            & (snr >= snr_min)
        )

        marker = markers.get(band, "o")

        # ZTF detections
        ax.errorbar(
            phase[mask],
            flux[mask],
            yerr=flux_err[mask],
            fmt=marker,
            label=band,
            color=colors[band],
        )

        # ZTF limits
        lim_flux = sn_df.loc[band_mask, "lim_flux_mJy"].values
        lim_mask = np.isfinite(phase) & np.isfinite(lim_flux) & (lim_flux > 0)
        ax.scatter(phase[lim_mask], lim_flux[lim_mask], marker="v", color=colors[band], alpha=0.5)

        # Keep track of x-limits from ZTF data
        phase_max = phase.max()+30
        xlim_ztf = ax.get_xlim()

    if show_ls:

        if ['ls_alpha'] not in sn_df.columns or ['ls_beta'] not in sn_df.columns:
            print("LS fit parameters not found in dataframe. Run fit_ls first.")

        # LS parameters 
        a_ls = sn_df["ls_alpha"].values[0]
        b_ls = sn_df["ls_beta"].values[0]

        t_grid = np.linspace(0, phase_max, 200)

        logf_ls = a_ls + b_ls * t_grid
        flux_ls = 10**logf_ls

        ax.plot(
                t_grid,
                flux_ls,
                color="mediumorchid",
                linestyle="--",
                label=f"LS: log F = {a_ls:.2f} + {b_ls:.4f} t",
                )
    
    if show_hbm:
        alpha_med = pred_df['alpha_med'].values[0]
        beta_med = pred_df['beta_med'].values[0]

        ax.plot(pred_df['phase'].values, pred_df['flux_q50_mJy'].values,
                color="mediumorchid",
                linewidth=2,
                label="HBM: log F = {alpha_med:.2f} + {beta_med:.4f} t")
        
        ax.fill_between(
            pred_df["phase"].values,
            pred_df["flux_q16_mJy"].values,
            pred_df["flux_q84_mJy"].values,
            color="mediumorchid",
            alpha=0.2,
            label="HBM: 16–84%",
        )

    if plot_wise:
        res_wise = get_wise_lc_data(oid, plot_LC=False)

        colors = {1: "navy", 2: "dodgerblue"}
        markers_w = {1: "s", 2: "s"}

        for fid, data in zip(
            [1, 2],
            [("b1_times", "b1_fluxes", "b1_fluxerrs"),
             ("b2_times", "b2_fluxes", "b2_fluxerrs")],
        ):
            color = colors.get(fid, "black")
            marker = markers_w.get(fid, "o")

            mask_det = ~np.isnan(res_wise[data[1]])
            if np.any(mask_det):
                ax.errorbar(
                    res_wise[data[0]][mask_det] - t0,
                    res_wise[data[1]][mask_det],
                    yerr=res_wise[data[2]][mask_det],
                    fmt=marker,
                    color=color,
                    label=f"W{fid}",
                )

    for col in ax.collections:
        if isinstance(col, mcoll.PathCollection):
            fcs = col.get_facecolors()
            if fcs is None or len(fcs) == 0:
                continue
            r, g, b, _ = fcs[0]
            col.set_facecolor((r, g, b, 0.3))
            col.set_edgecolor((r, g, b, 0.3))
            col.set_linewidth(1.2)
            col.set_sizes([20])

    for line in ax.lines:
        mfc = line.get_markerfacecolor()
        mec = line.get_markeredgecolor()
        if mfc is None or mfc == "none":
            continue

        if isinstance(mfc, (tuple, list)) and len(mfc) == 4:
            r, g, b, _ = mfc
        elif isinstance(mfc, str):
            r, g, b, _ = mcolors.to_rgba(mfc)
        else:
            continue
        line.set_markerfacecolor((r, g, b, 0.3))
        line.set_markeredgecolor((r, g, b, 1.0))
        line.set_markeredgewidth(1.2)
        line.set_markersize(7)

    ax.set_yscale("log")
    ax.set_xlabel("Phase (days)")
    ax.set_ylabel("Flux (mJy)")
    ax.set_title(f"Tail LC for {oid}")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

    if xlim != (None, None):
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-10, phase_max)

    if ylim != (None, None):
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()