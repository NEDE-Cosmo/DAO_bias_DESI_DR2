import math
import os
from pathlib import Path

import numpy as np

from cobaya.likelihood import Likelihood


clight = 299792.458  # km / s


class DAO_bias_DESI_DR2(Likelihood):
    """
    DESI DR2 BAO likelihood with DAO-induced alpha shifts.

    The input data stay in Q = {DM, DH, DV} / r_s, while a companion Q_fid file
    provides the fiducial values used to convert the mean vector and covariance
    to alpha-basis.

    The DAO shift follows
      alpha_parallel = (DH / r_B) / (DH / r_B)^fid
      alpha_perp     = (DM / r_B) / (DM / r_B)^fid
      alpha_iso      = (alpha_parallel alpha_perp^2)^(1/3)

      Sigma_vis^2  = Sigma_0^2 D^2 [1 + mu^2 f (2 + f)] + Sigma_Silk^2
      Sigma_dark^2 = Sigma_D0^2 D^2 [1 + mu^2 f (2 + f)] + Sigma_D,Silk^2

      (Delta alpha_parallel, Delta alpha_perp)^T = - Drr * Atilde * N^{-1} u
    """

    def initialize(self):
        if getattr(self, "data_directory", None) in (None, ""):
            self.data_directory = str(Path(__file__).resolve().parent / "data")
        if getattr(self, "data_file", None) in (None, ""):
            self.data_file = "desi_gaussian_bao_ALL_GCcomb_mean.txt"
        if getattr(self, "cov_file", None) in (None, ""):
            self.cov_file = "desi_gaussian_bao_ALL_GCcomb_cov.txt"
        if getattr(self, "qfid_file", None) in (None, ""):
            self.qfid_file = "desi_gaussian_bao_ALL_GCcomb_mean_qfid.txt"

        self.mean_path = self._join_data_path(self.data_file)
        self.cov_path = self._join_data_path(self.cov_file)
        self.qfid_path = self._join_data_path(self.qfid_file)

        self.z, q_obs, self.qtags = self._load_tagged_values(self.mean_path, "Mean file")
        cov = np.loadtxt(self.cov_path)
        if cov.shape != (len(self.z), len(self.z)):
            raise ValueError(
                "Covariance shape %s does not match data length %d"
                % (cov.shape, len(self.z))
            )

        self.qfid_by_z = self._load_qfid_file(self.qfid_path)
        fid_vals = np.array(
            [self._fiducial_quantity_over_rd(float(z), tag) for z, tag in zip(self.z, self.qtags)]
        )

        self.alpha_obs = q_obs / fid_vals
        scale = np.diag(1.0 / fid_vals)
        self.cov_alpha = scale @ cov @ scale
        self.cov_inv = np.linalg.inv(self.cov_alpha)

        self.n_data = len(self.z)
        self.mu_nodes, self.mu_weights = np.polynomial.legendre.leggauss(int(self.mu_npts))
        self.z_unique = [float(z) for z in self.z]
        self.z_background = sorted(set(float(z) for z in self.z) | {0.0})

    def get_requirements(self):
        return {
            "Hubble": {"z": self.z_background},
            "angular_diameter_distance": {"z": self.z_unique},
            "rs_drag": None,
            "z_d": None,
            "CLASS_background": None,
            "CLASS_thermodynamics": None,
        }

    def logp(self, **params_values):
        hubble_vals = np.atleast_1d(self.provider.get_Hubble(self.z_background, units="km/s/Mpc"))
        da_vals = np.atleast_1d(self.provider.get_angular_diameter_distance(self.z_unique))
        background = self.provider.get_CLASS_background()
        thermodynamics = self.provider.get_CLASS_thermodynamics()

        hubble_by_z = {float(z): float(val) for z, val in zip(self.z_background, hubble_vals)}
        da_by_z = {float(z): float(val) for z, val in zip(self.z_unique, da_vals)}

        h = hubble_by_z[0.0] / 100.0
        rd = float(self.provider.get_param("rs_drag"))
        z_drag = float(self.provider.get_param("z_d"))
        hrB = h * rd

        Drr = float(params_values["Drr"])
        Atilde = float(params_values["Atilde"])
        hSigma0 = float(params_values["hSigma0"])
        hSigmaD0 = float(params_values["hSigmaD0"])
        hSigmaDsilk = float(params_values["hSigmaDsilk"])
        hSigmaSilk = self._visible_silk_scale(h, z_drag, thermodynamics)
        D_of_z, f_of_z = self._growth_tables(background)

        alpha_theory = np.empty(self.n_data)
        dump_rows = []

        for i, (z, tag) in enumerate(zip(self.z, self.qtags)):
            z = float(z)
            da = da_by_z[z]
            dm = (1.0 + z) * da
            dh = clight / hubble_by_z[z]
            alpha_parallel = (dh / rd) / self._fiducial_quantity_over_rd(z, "DH_over_rs")
            alpha_perp = (dm / rd) / self._fiducial_quantity_over_rd(z, "DM_over_rs")
            alpha_iso = (alpha_parallel * alpha_perp * alpha_perp) ** (1.0 / 3.0)

            D = float(np.interp(z, D_of_z[0], D_of_z[1]))
            f = float(np.interp(z, f_of_z[0], f_of_z[1]))
            delta_parallel, delta_perp = self._delta_alpha_vector(
                D, f, hrB, Drr, Atilde, hSigma0, hSigmaD0, hSigmaSilk, hSigmaDsilk
            )

            eps_parallel = 0.0 if alpha_parallel == 0.0 else delta_parallel / alpha_parallel
            eps_perp = 0.0 if alpha_perp == 0.0 else delta_perp / alpha_perp
            delta_iso = alpha_iso * (2.0 * eps_perp + eps_parallel) / 3.0

            alpha_parallel_shifted = alpha_parallel + delta_parallel
            alpha_perp_shifted = alpha_perp + delta_perp
            alpha_iso_shifted = alpha_iso + delta_iso

            if tag == "DH_over_rs":
                alpha_theory[i] = alpha_parallel_shifted
            elif tag == "DM_over_rs":
                alpha_theory[i] = alpha_perp_shifted
            else:
                alpha_theory[i] = alpha_iso_shifted

            dump_rows.append(
                {
                    "z": z,
                    "tag": tag,
                    "alpha_parallel_base": alpha_parallel,
                    "alpha_perp_base": alpha_perp,
                    "alpha_iso_base": alpha_iso,
                    "delta_alpha_parallel": delta_parallel,
                    "delta_alpha_perp": delta_perp,
                    "delta_alpha_iso": delta_iso,
                    "alpha_parallel_shifted": alpha_parallel_shifted,
                    "alpha_perp_shifted": alpha_perp_shifted,
                    "alpha_iso_shifted": alpha_iso_shifted,
                    "alpha_theory_row": alpha_theory[i],
                }
            )

        self._write_dump(dump_rows, hrB, Drr, Atilde, hSigmaSilk, hSigma0, hSigmaD0, hSigmaDsilk)
        residual = self.alpha_obs - alpha_theory
        chi2 = float(residual @ (self.cov_inv @ residual))
        return -0.5 * chi2

    def _join_data_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_directory, path)

    def _load_tagged_values(self, path, label):
        rows = []
        with open(path, "r") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    raise ValueError("%s must have three columns: z, value, quantity_tag" % label)
                rows.append((float(parts[0]), float(parts[1]), str(parts[2])))

        if not rows:
            raise ValueError("%s contains no data rows." % label)

        z = np.array([row[0] for row in rows], dtype=float)
        values = np.array([row[1] for row in rows], dtype=float)
        tags = np.array([row[2] for row in rows], dtype=str)
        return z, values, tags

    def _load_qfid_file(self, path):
        z, values, tags = self._load_tagged_values(path, "Q_fid file")
        qfid_by_z = {}
        for z_i, value_i, tag_i in zip(z, values, tags):
            z_key = float(z_i)
            if z_key not in qfid_by_z:
                qfid_by_z[z_key] = {}
            qfid_by_z[z_key][tag_i] = float(value_i)
        return qfid_by_z

    def _fiducial_quantity_over_rd(self, z, tag):
        return self.qfid_by_z[float(z)][tag]

    def _growth_tables(self, background):
        if "z" not in background or "gr.fac. D" not in background or "gr.fac. f" not in background:
            raise ValueError(
                "CLASS background table must contain 'z', 'gr.fac. D' and 'gr.fac. f'."
            )

        z = np.asarray(background["z"], dtype=float)
        D = np.asarray(background["gr.fac. D"], dtype=float)
        f = np.asarray(background["gr.fac. f"], dtype=float)

        if z[0] > z[-1]:
            z = z[::-1]
            D = D[::-1]
            f = f[::-1]

        D0 = float(np.interp(0.0, z, D))
        return (z, D / D0), (z, f)

    def _visible_silk_scale(self, h, z_drag, thermodynamics):
        z = np.asarray(thermodynamics["z"], dtype=float)
        rd = np.asarray(thermodynamics["r_d"], dtype=float)

        if z[0] > z[-1]:
            z = z[::-1]
            rd = rd[::-1]

        z_eval = float(np.clip(z_drag, z[0], z[-1]))
        sigma_silk = float(np.interp(z_eval, z, rd)) / (2.0 * math.pi)
        return h * sigma_silk

    def _mu_average(self, values):
        return 0.5 * np.sum(self.mu_weights * values)


# F: This is where all the physics happens. The formulas are directly as in the paper: eqs. (23),(24),(25).
    def _delta_alpha_vector(
        self,
        D,
        f,
        hrB,
        Drr,
        Atilde,
        hSigma0,
        hSigmaD0,
        hSigmaSilk,
        hSigmaDsilk,
    ):
        if Drr == 0.0 or Atilde == 0.0:
            return 0.0, 0.0

        mu = self.mu_nodes 
        mu2 = mu * mu
        one_minus_mu2 = 1.0 - mu2

        #F: checked. Note that we follow a convention where all lengths scales are in units of Mpc (rather than Mpc/h), so the h factors are included in the Sigma parameters although this is not always reflected in the notation.
        sigma_vis2 = hSigma0 * hSigma0 * D * D * (1.0 + mu2 * f * (2.0 + f)) + hSigmaSilk * hSigmaSilk
        sigma_dark2 = (
            hSigmaD0 * hSigmaD0 * D * D * (1.0 + mu2 * f * (2.0 + f))
            + hSigmaDsilk * hSigmaDsilk
        )
        sigma_tot2 = sigma_vis2 + sigma_dark2

        #F: checked. As before, this would better be called h_delta_r. But then h `drops out' in the kernel anyways.
        delta_r = hrB * Drr
        
        #F: checked.
        kernel = (
            sigma_tot2 ** (-2.5)
            * (1.0 - delta_r * delta_r / (6.0 * sigma_tot2))
            * np.exp(-delta_r * delta_r / (4.0 * sigma_tot2))
        )
        
        #F: checked.
        norm = (2.0 * sigma_vis2) ** (-2.5)

        #F: checked. 
        N11 = self._mu_average(mu2 * mu2 * norm)
        N12 = self._mu_average(mu2 * one_minus_mu2 * norm)
        N22 = self._mu_average(one_minus_mu2 * one_minus_mu2 * norm)
        
        #F: checked. 
        u_parallel = self._mu_average(mu2 * kernel)
        u_perp = self._mu_average(one_minus_mu2 * kernel)
        
        
        #F: checked. 
        det_N = N11 * N22 - N12 * N12
        prefactor = -Drr * Atilde / det_N
        
        #F: checked. 
        delta_parallel = prefactor * (N22 * u_parallel - N12 * u_perp)
        delta_perp = prefactor * (-N12 * u_parallel + N11 * u_perp)
        return float(delta_parallel), float(delta_perp)

    def _write_dump(self, dump_rows, hrB, Drr, Atilde, hSigmaSilk, hSigma0, hSigmaD0, hSigmaDsilk):
        if self.dump_theory_to in (None, ""):
            return

        with open(self.dump_theory_to, "w") as handle:
            handle.write(
                "# hrB=%.10f Drr=%.10f Atilde=%.10f hSigmaSilk_vis=%.6f "
                "hSigma0=%.6f hSigmaD0=%.6f hSigmaDsilk=%.6f\n"
                % (hrB, Drr, Atilde, hSigmaSilk, hSigma0, hSigmaD0, hSigmaDsilk)
            )
            handle.write(
                "z,tag,alpha_parallel_base,alpha_perp_base,alpha_iso_base,"
                "delta_alpha_parallel,delta_alpha_perp,delta_alpha_iso,"
                "alpha_parallel_shifted,alpha_perp_shifted,alpha_iso_shifted,"
                "alpha_theory_row\n"
            )
            for row in dump_rows:
                handle.write(
                    "%.6f,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n"
                    % (
                        row["z"],
                        row["tag"],
                        row["alpha_parallel_base"],
                        row["alpha_perp_base"],
                        row["alpha_iso_base"],
                        row["delta_alpha_parallel"],
                        row["delta_alpha_perp"],
                        row["delta_alpha_iso"],
                        row["alpha_parallel_shifted"],
                        row["alpha_perp_shifted"],
                        row["alpha_iso_shifted"],
                        row["alpha_theory_row"],
                    )
                )
