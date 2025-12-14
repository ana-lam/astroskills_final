import numpy as np

def fit_ls(phase, flux, flux_err, snr_min=3.0, min_points=3):
    """
    Fit a least-squares line to the given phase and flux data (log-flux).
    """

    phase = np.array(phase)
    flux = np.array(flux)
    flux_err = np.array(flux_err)

    mask = np.isfinite(phase) & np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0) & ((flux / flux_err) >= snr_min)

    if mask.sum() < min_points:
        print("Not enough points for fitting.")
        return None, None

    t = phase[mask]
    f = np.log10(flux[mask])
    f_err = flux_err[mask] / (flux[mask] * np.log(10))

    # weights from flux errors
    weights = 1.0 / (f_err ** 2)

    sum_w = np.sum(weights)
    sum_wt = np.sum(weights * t)
    sum_wf = np.sum(weights * f)
    sum_wtt = np.sum(weights * t * t)
    sum_wtf = np.sum(weights * t * f)

    # determinant
    delta = sum_w * sum_wtt - sum_wt ** 2
    if delta <= 0:
        print("Non-positive determinant in least-squares fitting.")
        return None, None
    if delta == 0:
        print("Zero determinant in least-squares fitting.")
        return None, None

    beta = (sum_w * sum_wtf - sum_wt * sum_wf) / delta
    alpha = (sum_wtt * sum_wf - sum_wt * sum_wtf) / delta

    var_beta = sum_w / delta
    sigma_beta = np.sqrt(var_beta)

    var_alpha = sum_wtt / delta
    sigma_alpha = np.sqrt(var_alpha)

    y_model = alpha + beta * t
    resid = f - y_model
    chi2 = np.sum((resid/f_err) ** 2)
    dof = len(t) - 2
    red_chi2 = chi2 / dof if dof > 0 else np.nan

    results = {
        "ls_alpha": alpha,
        "ls_beta": beta,
        "ls_sigma_alpha": sigma_alpha,
        "ls_sigma_beta": sigma_beta,
        "ls_chi2": chi2,
        "ls_red_chi2": red_chi2
    }

    return results, (chi2, red_chi2)
    