import os
import pandas as pd
import numpy as np
from alerce.core import Alerce
import pickle
from .plotting import plot_ztf_lc, plot_ztf_forced_lc
from pathlib import Path

# data directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# metadata df
meta = pd.read_csv(DATA_DIR / "zenodo_metasample.csv")

# params df
params = pd.read_csv(DATA_DIR / "TableA_full_machine_readable_params.csv")

# alerce client
client = Alerce()

def _convert_mag_to_flux(res, forced=True):
    """
    Convert ZTF magnitudes to mJy.
    """

    # Detections
    if "lc_det" in res and not res["lc_det"].empty:
        mask_det = res["lc_det"].magpsf.notna()
        mag = res["lc_det"].loc[mask_det, "magpsf"].values
        mag_err = res["lc_det"].loc[mask_det, "sigmapsf"].values

        flux_mJy = 3631 * 10**(-mag/2.5) * 1e3
        flux_err_mJy = flux_mJy * (np.log(10)/2.5) * mag_err

        res["lc_det"].loc[mask_det, "flux_mJy"] = flux_mJy
        res["lc_det"].loc[mask_det, "flux_err_mJy"] = flux_err_mJy 

    # Non-detections
    if "lc_nondet" in res and not res["lc_nondet"].empty:
        mask_nondet = res["lc_nondet"].diffmaglim > 0
        lim_mag = res["lc_nondet"].loc[mask_nondet, "diffmaglim"].values
        lim_flux_mJy = 3631 * 10**(-lim_mag/2.5) * 1e3
        res["lc_nondet"].loc[mask_nondet, "lim_flux_mJy"] = lim_flux_mJy

    # Do the same for forced photometry
    if forced and "forced" in res:
        for filt, data in res["forced"].items():
            if "flux_mJy" not in data:
                data["flux_mJy"] = np.full_like(data["mag"], np.nan, dtype=float)
                data["flux_err_mJy"] = np.full_like(data["mag"], np.nan, dtype=float)
                data["lim_flux_mJy"] = np.full_like(data["limiting_mag"], np.nan, dtype=float)

            # Detections
            mask_det = ~np.isnan(data["mag"])
            mag = data["mag"][mask_det]
            mag_err = data["mag_err"][mask_det]
            flux_mJy = 3631 * 10**(-mag/2.5) * 1e3
            flux_err_mJy = flux_mJy * (np.log(10)/2.5) * mag_err
            data["flux_mJy"][mask_det] = flux_mJy
            data["flux_err_mJy"][mask_det] = flux_err_mJy

            # Non-detections: prefer mag_ul, fallback to limiting_mag
            has_ul = "mag_ul" in data
            mask_nondet = np.isnan(data["mag"]) & (
                np.isfinite(data["mag_ul"]) if has_ul else (data["limiting_mag"] > 0)
            )
            if np.any(mask_nondet):
                lim_mag = (data["mag_ul"][mask_nondet] if has_ul
                           else data["limiting_mag"][mask_nondet])
                lim_flux_mJy = 3631 * 10**(-lim_mag/2.5) * 1e3
                data["lim_flux_mJy"][mask_nondet] = lim_flux_mJy
    return res 

def _get_ztf_forcedphot(forced_file, SNT=3.0, SNU=5.0): # SNT & SNU values from https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_forced_photometry.pdf
    """
    Parse a ZTF forced photometry .dat file into times, fluxes, errors, and filters.
    Returns a dictionary grouped by filter.
    """

    # Load into df, skipping header lines
    df = pd.read_csv(
        forced_file,
        comment='#',
        sep=r"\s+",
        header=0,
        skiprows=1
    )
    df.columns = df.columns.str.replace(",", "").str.strip()

    # Convert JD to MJD
    df["mjd"] = df["jd"] - 2400000.5

    # Compute SNR
    flux = df['forcediffimflux'].astype(float)
    flux_err = df['forcediffimfluxunc'].astype(float)
    snr = flux / flux_err

    # Detections
    det = (snr > SNT) & (flux > 0)

    # Compute magnitudes and errors for detections only
    # mag = zpdiff - 2.5*log10(flux), valid only if flux > 0
    df["mag"] = np.nan
    df["mag_err"] = np.nan
    df.loc[det, "mag"] = df.loc[det, "zpdiff"] - 2.5 * np.log10(df.loc[det, "forcediffimflux"])
    # 1.0857 = 2.5 / ln(10)
    df.loc[det, "mag_err"] = 1.0857 * df.loc[det, "forcediffimfluxunc"] / df.loc[det, "forcediffimflux"]

    # Non-detections (upper limits)
    # mag_UL = zpdiff - 2.5*log10(SNU * sigma_flux)
    non_det = ~det
    df['mag_ul'] = np.nan
    df.loc[non_det, "mag_ul"] = df.loc[non_det, "zpdiff"] - 2.5 * np.log10(SNU * df.loc[non_det, "forcediffimfluxunc"])

    # Organize output
    res = {}
    for filt in df["filter"].unique():
        mask = df["filter"] == filt
        res[filt] = {
            "mjd": df.loc[mask, "mjd"].values,
            "flux": df.loc[mask, "forcediffimflux"].values, # DN units
            "flux_err": df.loc[mask, "forcediffimfluxunc"].values, # DN units
            "snr": snr.loc[mask].values, 
            "mag": df.loc[mask, "mag"].values,
            "mag_err": df.loc[mask, "mag_err"].values,
            "mag_ul": df.loc[mask, "mag_ul"].values,
            "limiting_mag": df.loc[mask, "diffmaglim"].values,
        }
    return res

def get_ztf_lc_data(oid, ra=None, dec=None, plot_LC=False, plot_flux=False,
                    add_forced=True, pad_before=100, pad_after=600):
    
    """
    Fetch detections, non-detections, and optionally IRSA forced photometry for a ZTF object.

    Parameters
    ----------
    oid : str
        Object ID.
    ra, dec : float, optional
        Sky coordinates (deg, ICRS). Required if add_forced=True.
    plot_LC : bool, default False
        Whether to plot ALeRCE light curve.
    doStamps : bool, default False
        Whether to fetch and plot image stamps.
    add_forced : bool, default True
        Add Das forced photometry.
    pad_before, pad_after : int
        Days before and after the first detection to include in forced photometry request.
    """

    # Save in pkl because sometimes Alerce API is down
    pkl_filename = DATA_DIR / f"ztf_resdicts/{oid}.pkl"

    results = {"oid": oid}

    det_ok = False
    nondet_ok = False

    # -- Fetch ALeRCE detections --
    try:
        lc_det = client.query_detections(oid, format='pandas').sort_values("mjd")
        results["lc_det"] = lc_det
        det_ok = True
    except Exception as e:
        print(f"Could not fetch detections for {oid}: {e}")

    # -- Fetch ALeRCE non-detections --
    try:
        lc_nondet = client.query_non_detections(oid, format='pandas').sort_values("mjd")
        results["lc_nondet"] = lc_nondet
        nondet_ok = True
    except Exception as e:
        print(f"Could not fetch non-detections for {oid}: {e}")
        lc_nondet = pd.DataFrame()

    if not det_ok or not nondet_ok:
        if pkl_filename.exists():
            try:
                with open(pkl_filename, "rb") as f:
                    cached = pickle.load(f)
                print(f"Loaded cached ZTF data for {oid}")
                return cached
            except Exception as e:
                raise RuntimeError(
                    f"API failed and cached file could not be loaded for {oid}: {e}"
                )
        else:
            raise RuntimeError(f"API failed and no cached file exists for {oid}")

    if add_forced:
        # Use Kaustav's zenodo published forced photometry data
        forced_file = DATA_DIR / f"Das_forced_photometry_files/{oid}_fps.dat"
        res_forced = _get_ztf_forcedphot(forced_file)
        results['forced'] = res_forced

    results = _convert_mag_to_flux(results, forced=add_forced)
        
    if plot_LC:
        if add_forced:
            plot_ztf_forced_lc(results['forced'], oid, flux=plot_flux)
        else:
            plot_ztf_lc(results["lc_det"], results["lc_nondet"], oid, flux=plot_flux)

    with open(pkl_filename, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results