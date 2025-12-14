from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from .ztf import get_ztf_lc_data
from .wise import get_wise_lc_data
from .misc import fit_ls
from .plotting import plot_tail_models

# data directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# params df
params = pd.read_csv(DATA_DIR / "TableA_full_machine_readable_params.csv")

class TailHBModel:
    """Hierarchical Bayesian Model for Type II Radioactive Tail Inference"""
    def __init__(self, params_df=params, band_use="ZTF_r", 
                 SNR_cut=3.0, normal_bounds=(16, 84)):
        self.params_df = params_df
        self.band_use = band_use 
        self.SNR_cut = SNR_cut # minimum SNR for data points
        self.normal_bounds = normal_bounds # percentile bounds for normal sample

        # set with prepare_data()
        self.df_clean = None

        self.df_det = None
        self.x = None
        self.y = None
        self.y_err = None
        self.sn_idx = None
        self.N_SN = None

        # PyMC model
        self.model = None
        self.inf_data = None

        # initial guesses
        self.init_mu_alpha = None
        self.init_sigma_alpha = None
        self.init_mu_beta = None
        self.init_sigma_beta = None

        # posterior predictives
        self.pred_curves = {}
        self.pred_curves_ppc = {}

    def _get_tail_start(self, oid, buffer_days=10.0):
        """Get the start time, t0,  of the tail for a specific object."""
        row = self.params_df[self.params_df['name'] == oid]
        if row.empty:
            print(f"No parameters found for {oid}")
            return None
        row = row.iloc[0]
        ts = []
        if not pd.isna(row['tailstart']):
            ts.append(row['tailstart'])
        if not pd.isna(row['plateauend']):
            t_plat = row['plateauend'] + buffer_days
            ts.append(t_plat)

        t0 = np.min(ts) if ts else None

        return t0 + buffer_days

    def _get_tail_drop(self, oid):
        row = self.params_df[self.params_df['name'] == oid]
        if row.empty:
            print(f"No parameters found for {oid}")
            return None
        row = row.iloc[0]
        delta_mag = row['plateauendmag']-row['tailstartmag']

        return delta_mag
    
    def _extract_tail_points_for_sn(self, oid, min_phase=0.0,
                                     max_phase=300.0):
        """Extract tail points for a specific SN."""
        BANDS = ['ZTF_g', 'ZTF_r', 'ZTF_i']
        ztf_res = get_ztf_lc_data(oid, add_forced=True, plot_LC=False)
        if "forced" not in ztf_res:
            print("No forced")
            return pd.DataFrame()

        t_tail = self._get_tail_start(oid)
        delta_mag = self._get_tail_drop(oid)

        if t_tail is None or not np.isfinite(t_tail):
            print("t_tail is None")
            return pd.DataFrame()

        rows = []
        ztf_forced = ztf_res['forced']

        for band in BANDS:
            if band not in ztf_forced:
                print(f"No data for band {band}")
                continue
            data = ztf_forced[band]
            mjd = np.array(data['mjd'])
            flux = np.array(data['flux_mJy'])
            flux_err = np.array(data['flux_err_mJy'])
            lim_flux = np.array(data['lim_flux_mJy'])

            mask = np.isfinite(mjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
            mjd, flux, flux_err = mjd[mask], flux[mask], flux_err[mask]

            phase = mjd - t_tail

            in_tail = (phase >= min_phase) & (phase <= max_phase) & (flux/flux_err >= self.SNR_cut) & (flux > 0)

            mjd, phase, flux, flux_err = mjd[in_tail], phase[in_tail], flux[in_tail], flux_err[in_tail]

            for mjd_i, phase_i, flux_i, flux_err_i, lim_flux_i in zip(mjd, phase, flux, flux_err, lim_flux):
                rows.append({
                    'oid': oid,
                    'delta_mag': delta_mag,
                    'band': band,
                    'mjd': mjd_i,
                    't0': t_tail,
                    'phase': phase_i,
                    'flux_mJy': flux_i,
                    'flux_err_mJy': flux_err_i,
                    'lim_flux_mJy': np.nan
                })

            mjd = np.array(data['mjd'])
            lim_flux = np.array(data['lim_flux_mJy'])
            lim_mask = np.isfinite(mjd) & np.isfinite(lim_flux) & (lim_flux > 0)
            mjd, lim_flux = mjd[lim_mask], lim_flux[lim_mask]

            phase = mjd - t_tail

            in_tail = (phase >= min_phase) & (phase <= max_phase) & (lim_flux > 0)
            mjd_lim, phase_lim, lim_flux = mjd[in_tail], phase[in_tail], lim_flux[in_tail]
    
            for mjd_i, phase_i, lim_flux_i in zip(mjd_lim, phase_lim, lim_flux):
                rows.append({
                    'oid': oid,
                    'delta_mag': delta_mag,
                    'band': band,
                    'mjd': mjd_i,
                    't0': t_tail,
                    'phase': phase_i,
                    'flux_mJy': np.nan,
                    'flux_err_mJy': np.nan,
                    'lim_flux_mJy': lim_flux_i
                })

        return pd.DataFrame(rows)

    def _build_tail_dataset(self, oids):
        """Build a dataset of tail points for a list of OIDs."""
        dfs = []
        for oid in oids:
            df_sn = self._extract_tail_points_for_sn(oid)
            if not df_sn.empty:
                dfs.append(df_sn)

        if len(dfs) == 0:
            print("No data found for any SNe.")
            return pd.DataFrame(), {}, {}
        df_all = pd.concat(dfs, ignore_index=True)

        sn_ids = {oid: i for i, oid in enumerate(df_all["oid"].unique())}
        band_ids = {b: i for i, b in enumerate(sorted(df_all["band"].unique()))}
        df_all["sn_idx"] = df_all["oid"].map(sn_ids)
        df_all["band_idx"] = df_all["band"].map(band_ids)

        return df_all, sn_ids, band_ids

    def _fit_ls_dataset(self, df, min_points=3):
        """Fit least-squares lines to each SN in the dataset."""
        df = df[df['band']==self.band_use].copy()
        rows = []
        for oid, sn_df in df.groupby('oid'):
            phase = sn_df['phase'].values
            flux = sn_df['flux_mJy'].values
            flux_err = sn_df['flux_err_mJy'].values

            res, fit_stats = fit_ls(phase, flux, flux_err, snr_min=self.SNR_cut, min_points=min_points)
            if not isinstance(res, dict):
                continue

            res_row = {'oid': oid}
            res_row.update(res)
            rows.append(res_row)
        
        ls_summary = pd.DataFrame(rows)

        return ls_summary

    def clean_data(self):
        """Prepare data for modeling"""
        # first grab SNe with WISE data
        self.params_df['WISE_det'] = False

        for idx, row in self.params_df.iterrows():
            oid = row['name']
            wise_res = get_wise_lc_data(oid, plot_LC=False)
            if wise_res != {} and len(wise_res.get('b1_times', [])) + len(wise_res.get('b2_times', [])) > 0:
                self.params_df.at[idx, 'WISE_det'] = True

        self.params_df = self.params_df[self.params_df['WISE_det']==True]

        self.params_df = self.params_df.drop_duplicates(subset='name', keep='first').reset_index(drop=True)

        self.df_clean, sn_ids, band_ids = self._build_tail_dataset(self.params_df['name'].tolist())

        # filter normal sample
        df_sort = self.df_clean.sort_values(by='delta_mag')
        df_dedupped = df_sort.drop_duplicates(subset='oid')
        oids = df_dedupped['oid'].to_numpy()
        delta_mags = df_dedupped['delta_mag'].to_numpy()
        med = np.median(delta_mags)
        lower_bound, upper_bound = np.percentile(delta_mags, self.normal_bounds)

        normal_mask = (delta_mags >= lower_bound) & (delta_mags <= upper_bound)

        anomalous_mask = ~normal_mask
        normal_oids = oids[normal_mask]
        anomalous_oids = oids[anomalous_mask]

        self.df_clean['normal_sample'] = False
        for idx, row in self.df_clean.iterrows():
            oid = row['oid']
            delta_mag = row['delta_mag']
            if (delta_mag >= lower_bound) & (delta_mag <= upper_bound):
                self.df_clean.at[idx, 'normal_sample'] = True

        self.df_clean = self.df_clean[self.df_clean['normal_sample']==True].reset_index(drop=True)

        # grab ls fits
        fit_summary = self._fit_ls_dataset(self.df_clean, min_points=3)
        self.df_clean = self.df_clean.merge(
            fit_summary, on='oid', how='left'
        )

        return self.df_clean

    def prepare_data(self):
        """Prepare data for modeling."""
        if self.df_clean is None:
            self.clean_data()

        df_model = self.df_clean[self.df_clean['band']==self.band_use].copy()

        mask_det = (
            (df_model['flux_mJy'].notna() & (df_model['flux_mJy'] > 0) &
            (df_model['flux_err_mJy'].notna()) & (df_model['flux_err_mJy'] > 0) & 
            ((df_model['flux_mJy'] / df_model['flux_err_mJy']) >= self.SNR_cut))
        )

        self.df_det = df_model[mask_det].copy()

        # Reindex SN IDs to be contiguous for the modeled subset
        codes, uniques = pd.factorize(self.df_det["oid"], sort=True)
        self.df_det["sn_idx_model"] = codes
        self.sn_idx = self.df_det["sn_idx_model"].to_numpy()
        self.N_SN = len(uniques)

        self.x = self.df_det['phase'].values
        self.y = np.log10(self.df_det['flux_mJy'].values)
        self.y_err = self.df_det['flux_err_mJy'].values / (self.df_det['flux_mJy'].values * np.log(10))
        # self.sn_idx = self.df_det['sn_idx'].values
        # self.N_SN = self.df_det['sn_idx'].nunique()

        df_r_unique = (
            self.df_clean[(self.df_clean["band"] == self.band_use) & self.df_clean["ls_alpha"].notna() & self.df_clean["ls_beta"].notna()] 
            .drop_duplicates(subset="oid")
            .copy()
        )

        alpha_ls = df_r_unique["ls_alpha"].values
        beta_ls  = df_r_unique["ls_beta"].values

        self.init_mu_alpha = np.mean(alpha_ls)
        self.init_sigma_alpha = np.std(alpha_ls)

        self.init_mu_beta = np.mean(beta_ls)
        self.init_sigma_beta = np.std(beta_ls)

        return self.x, self.y, self.y_err, self.sn_idx

    def build_model(self):
        """Build the PyMC model."""
        if self.x is None or self.y is None or self.y_err is None or self.sn_idx is None:
            self.prepare_data()

        with pm.Model() as model:
            # hyperpriors for population parameters
            mu_alpha = pm.Normal("mu_alpha", mu=self.init_mu_alpha, sigma=1.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=self.init_sigma_alpha)

            mu_beta = pm.Normal("mu_beta", mu=self.init_mu_beta, sigma=0.1)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=self.init_sigma_beta)

            # intrinsic scatter (beyond measurement errors)
            sigma_int = pm.HalfNormal("sigma_int", sigma=0.1)

            # SN specific parameters
            # alpha and beta for each SN
            alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=self.N_SN)
            beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=self.N_SN)

            # observational data
            mu_y = alpha[self.sn_idx] + beta[self.sn_idx] * self.x
            total_sigma = pm.math.sqrt(self.y_err**2 + sigma_int**2)

            y_obs = pm.Normal("y_obs", mu=mu_y, sigma=total_sigma, observed=self.y)
        
        self.model = model 

    def fit_model(self, draws=2000, tune=500, chains=8, cores=4, target_accept=0.9,
                  random_seed=42, save_filename=None):
        """Fit the PyMC model."""
        if self.model is None:
            self.build_model()

        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True
            )
        if save_filename is not None:
            az.to_netcdf(trace, DATA_DIR / f"fits/{save_filename}")
            print("Saved inference data to", DATA_DIR / f"fits/{save_filename}")
        else:
            az.to_netcdf(trace, DATA_DIR / "fits/tail_hbm_inference.nc")
            print("Saved inference data to", DATA_DIR / "fits/tail_hbm_inference.nc")
        self.inf_data = trace
        return trace
    
    def predict_tail(self, oid, phases=None, q=(16, 50, 84), include_intrinsic_scatter=False,
                     random_seed=42):
        """Predict tail fluxes for a specific SN at given phases."""

        if self.inf_data is None:
            print("Model has not been fit yet.")
            return None
        
        if self.df_det is None or "sn_idx_model" not in self.df_det.columns:
            self.prepare_data()

        if phases is None:
            phases = np.linspace(-10, 330., 311)
        
        oid_to_idx = dict(zip(self.df_det["oid"], self.df_det["sn_idx_model"]))

        if oid not in oid_to_idx:
            print(f"OID {oid} not in fitted data.")
            return None
        
        i = int(oid_to_idx[oid])

        # pull posterior samples
        posterior = self.inf_data.posterior
        alpha_samples = posterior["alpha"].isel(alpha_dim_0=i).stack(s=("chain", "draw")).values
        beta_samples = posterior["beta"].isel(beta_dim_0=i).stack(s=("chain", "draw")).values
        sigma_int_samples = posterior["sigma_int"].stack(s=("chain", "draw")).values

        mu_log = alpha_samples[:, None] + beta_samples[:, None] * phases[None, :]

        if include_intrinsic_scatter:
            rng = np.random.default_rng(random_seed)
            eps = rng.normal(0, sigma_int_samples[:, None], size=mu_log.shape)
            y_log = mu_log + eps
        else:
            y_log = mu_log

        df = pd.DataFrame({"phase": phases})
        for qi in q:
            q_vals = np.percentile(y_log, qi, axis=0)
            df[f"flux_log_q{qi}"] = q_vals
            df[f"flux_q{qi}_mJy"] = 10**q_vals

        df['oid'] = oid
        df['band'] = self.band_use
        df['kind'] = 'ppc' if include_intrinsic_scatter else 'mean'

        alpha_med = float(np.median(alpha_samples))
        beta_med  = float(np.median(beta_samples))
        sigma_int_med = float(np.median(sigma_int_samples))
        
        df['alpha_med'] = alpha_med
        df['beta_med'] = beta_med
        df['sigma_int_med'] = sigma_int_med

        if include_intrinsic_scatter:
            self.pred_curves_ppc[oid] = df
        else:
            self.pred_curves[oid] = df

        return df
