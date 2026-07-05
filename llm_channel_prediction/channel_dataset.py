"""Synthetic ISAC channel dataset.

Implements the geometric channel model of Sec. II, Eqs. (1)-(2) of
arXiv:2505.09141: a multi-antenna OFDM ISAC BS with a UPA of N = Nv x Nh
elements serves a single-antenna mobile user (MU).  Scatterers are split
into *shared*, *communication-only* and *sensing-only* sets; shared
scatterers contribute paths with identical Doppler shift and departure
angles {f_d,i, theta_i, phi_i} to both the bi-static communication channel
h_c,t (Eq. (1)) and the mono-static sensing channel H_s,t (Eq. (2)), which
is exactly the correlation the predictor is meant to exploit.

The paper generates its dataset with NVIDIA Sionna ray tracing in a street
scene.  Ray tracing is not reproducible from the letter alone (the scene
geometry is not published), so this module draws the per-path parameters
{alpha_i, beta_i, tau_i, f_d,i, theta_i, phi_i} at random with the same
statistics the paper reports (path counts from Fig. 2(b), MU speed range,
quasi-stationary paths over the P+Q slots) and then evaluates Eqs. (1)-(2)
exactly.  Channel variation across slots therefore comes from the
constructive/destructive combination of many random-phase paths, as the
paper assumes (Sec. II, discussion below Eq. (1)).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SystemConfig


# --------------------------------------------------------------------------
# geometric channel generation
# --------------------------------------------------------------------------

def steering_matrix(theta: np.ndarray, phi: np.ndarray,
                    n_v: int, n_h: int) -> np.ndarray:
    """UPA steering vectors a(theta, phi) = a_v(theta) (x) a_h(theta, phi)
    for half-wavelength spacing (d_v = d_h = lambda / 2).

    Parameters are arrays of shape (n_paths,); returns (n_paths, N)."""
    theta = theta[:, None]
    phi = phi[:, None]
    m_v = np.arange(n_v)[None, :]
    m_h = np.arange(n_h)[None, :]
    # d/lambda = 1/2  =>  2*pi*d/lambda = pi
    a_v = np.exp(1j * np.pi * m_v * np.sin(theta)) / np.sqrt(n_v)
    a_h = np.exp(1j * np.pi * m_h * np.cos(theta) * np.sin(phi)) / np.sqrt(n_h)
    # Kronecker product per path: (n_paths, Nv*Nh)
    return (a_v[:, :, None] * a_h[:, None, :]).reshape(theta.shape[0], -1)


def _draw_paths(rng: np.random.Generator, n_paths: int, cfg: SystemConfig,
                speed_mps: float):
    """Draw random per-path parameters {theta, phi, tau, gain, f_d}."""
    theta = np.deg2rad(rng.uniform(-cfg.elevation_range, cfg.elevation_range,
                                   n_paths))
    phi = np.deg2rad(rng.uniform(-cfg.azimuth_range, cfg.azimuth_range,
                                 n_paths))
    tau = rng.uniform(0.0, cfg.tau_max, n_paths)
    # exponential power-delay profile with random uniform phases
    power = np.exp(-tau / cfg.pdp_decay)
    power /= power.sum()
    gain = np.sqrt(power) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_paths))
    # Doppler: contribution of the MU motion plus the motion of the
    # (vehicle) scatterer itself, each with a random direction cosine
    v_sc = cfg.v_scatter_max_kmh / 3.6
    f_d = (speed_mps / cfg.wavelength) * rng.uniform(-1, 1, n_paths) \
        + (v_sc / cfg.wavelength) * rng.uniform(-1, 1, n_paths)
    return theta, phi, tau, gain, f_d


def generate_sample(rng: np.random.Generator, cfg: SystemConfig,
                    speed_kmh: float):
    """Generate one dataset sample: the communication CSI of all P+Q slots
    and the sensing CSI of the P historical slots.

    Returns
    -------
    comm : complex64, (N, K, P+Q)
        Frequency-domain communication CSI  h_bar_c,t[k]  per antenna.
    sens : complex64, (N, K, P)
        Diagonal of the frequency-domain sensing CSI  H_bar_s,t[k]
        per antenna (the paper's per-antenna parallel processing uses the
        n-th diagonal element, Sec. III-A).
    """
    speed = speed_kmh / 3.6
    K, N = cfg.n_subcarriers, cfg.n_antennas
    T = cfg.n_total_slots
    t_slots = np.arange(T) * cfg.slot_duration
    f_sub = np.arange(K) * cfg.subcarrier_spacing

    # ---- shared scatterers: same {f_d, theta, phi} for comm & sensing ----
    th_sh, ph_sh, tau_sh, g_sh, fd_sh = _draw_paths(
        rng, cfg.n_shared, cfg, speed)
    # ---- communication-only scatterers ----
    th_c, ph_c, tau_c, g_c, fd_c = _draw_paths(
        rng, cfg.n_comm_only, cfg, speed)
    # ---- sensing-only scatterers ----
    th_s, ph_s, tau_s, g_s, fd_s = _draw_paths(
        rng, cfg.n_sens_only, cfg, speed)

    # ---- communication channel, Eq. (1): shared + comm-only paths ----
    theta = np.concatenate([th_sh, th_c])
    phi = np.concatenate([ph_sh, ph_c])
    tau = np.concatenate([tau_sh, tau_c])
    gain = np.concatenate([g_sh, g_c])
    f_d = np.concatenate([fd_sh, fd_c])
    A = steering_matrix(theta, phi, cfg.n_v, cfg.n_h)          # (Pc, N)
    doppler = np.exp(1j * 2 * np.pi * f_d[:, None] * t_slots)  # (Pc, T)
    delay = np.exp(-1j * 2 * np.pi * tau[:, None] * f_sub)     # (Pc, K)
    comm = np.einsum("pn,pk,pt->nkt", gain[:, None] * A, delay, doppler)

    # ---- sensing channel, Eq. (2): shared + sensing-only paths ----
    # Shared paths keep the same {f_d, theta, phi} but have their own
    # complex round-trip coefficient beta_i and round-trip delay tau_bar_i.
    tau_sh_rt = tau_sh * rng.uniform(1.2, 2.0, cfg.n_shared)
    beta_sh = np.abs(g_sh) * np.exp(1j * rng.uniform(0, 2 * np.pi,
                                                     cfg.n_shared))
    theta2 = np.concatenate([th_sh, th_s])
    phi2 = np.concatenate([ph_sh, ph_s])
    tau2 = np.concatenate([tau_sh_rt, tau_s])
    beta = np.concatenate([beta_sh, g_s])
    f_d2 = np.concatenate([fd_sh, fd_s])
    A2 = steering_matrix(theta2, phi2, cfg.n_v, cfg.n_h)       # (Ps, N)
    # per-antenna sensing observation = n-th diagonal of a a^T  =  a_n^2
    A2_diag = A2 ** 2
    doppler2 = np.exp(1j * 2 * np.pi * f_d2[:, None]
                      * t_slots[:cfg.n_hist])                  # (Ps, P)
    delay2 = np.exp(-1j * 2 * np.pi * tau2[:, None] * f_sub)   # (Ps, K)
    sens = np.einsum("pn,pk,pt->nkt", beta[:, None] * A2_diag,
                     delay2, doppler2)

    return comm.astype(np.complex64), sens.astype(np.complex64)


def generate_dataset(cfg: SystemConfig, n_samples: int, seed: int,
                     speed_kmh: float | None = None):
    """Generate ``n_samples`` channel realizations.

    speed_kmh=None draws the MU speed uniformly in
    [cfg.speed_min_kmh, cfg.speed_max_kmh] per sample (training / mixed
    test set); a float fixes the speed (used for the Fig. 4 sweep)."""
    rng = np.random.default_rng(seed)
    N, K = cfg.n_antennas, cfg.n_subcarriers
    comm = np.empty((n_samples, N, K, cfg.n_total_slots), np.complex64)
    sens = np.empty((n_samples, N, K, cfg.n_hist), np.complex64)
    speeds = np.empty(n_samples, np.float32)
    for s in range(n_samples):
        v = (rng.uniform(cfg.speed_min_kmh, cfg.speed_max_kmh)
             if speed_kmh is None else speed_kmh)
        speeds[s] = v
        comm[s], sens[s] = generate_sample(rng, cfg, v)
    return comm, sens, speeds


# --------------------------------------------------------------------------
# torch dataset (preprocessor module of Sec. III-A)
# --------------------------------------------------------------------------

def _to_real(x: np.ndarray) -> np.ndarray:
    """(K, T) complex -> (2, K, T) float32 (real / imaginary parts)."""
    return np.stack([x.real, x.imag]).astype(np.float32)


def _normalize(x: np.ndarray):
    """Zero-mean unit-std normalization; returns (x_norm, mean, std)."""
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std, np.float32(mean), np.float32(std)


class CsiDataset(Dataset):
    """One item = the CSI of one (sample, antenna) pair, matching the
    paper's per-antenna parallel processing (Sec. III-A).

    Implements the preprocessor module: K-point IDFT to the delay domain,
    complex -> real conversion, and per-item normalization.  De-norm
    statistics of the frequency-domain communication CSI are returned so
    the network output can be mapped back (Eq. (7)).

    ``snr_db`` adds white Gaussian noise to the *historical* CSI (both
    communication and sensing) with SNR = 1/sigma_n^2 relative to the CSI
    power, as in the paper's Fig. 5 experiment.  A (lo, hi) tuple draws
    the SNR uniformly per item (noise-augmented training)."""

    def __init__(self, comm: np.ndarray, sens: np.ndarray, cfg: SystemConfig,
                 snr_db=None, max_antennas: int = 0, seed: int = 0):
        self.comm = comm
        self.sens = sens
        self.cfg = cfg
        self.snr_db = snr_db
        self.n_ant = comm.shape[1] if max_antennas <= 0 \
            else min(max_antennas, comm.shape[1])
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.comm.shape[0] * self.n_ant

    def _add_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        p_sig = np.mean(np.abs(x) ** 2)
        sigma2 = p_sig * 10.0 ** (-snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2) * (
            self.rng.standard_normal(x.shape)
            + 1j * self.rng.standard_normal(x.shape))
        return (x + noise).astype(np.complex64)

    def __getitem__(self, idx):
        s, n = divmod(idx, self.n_ant)
        P = self.cfg.n_hist
        comm_hist = self.comm[s, n, :, :P]          # (K, P) complex
        target = self.comm[s, n, :, P:]             # (K, Q) complex
        sens_hist = self.sens[s, n]                 # (K, P) complex

        if self.snr_db is not None:
            snr = (self.rng.uniform(*self.snr_db)
                   if isinstance(self.snr_db, (tuple, list))
                   else float(self.snr_db))
            comm_hist = self._add_noise(comm_hist, snr)
            sens_hist = self._add_noise(sens_hist, snr)

        # K-point IDFT: frequency -> delay domain (preprocessor, Sec. III-A)
        comm_delay = np.fft.ifft(comm_hist, axis=0)
        sens_delay = np.fft.ifft(sens_hist, axis=0)

        comm_f, mean_c, std_c = _normalize(_to_real(comm_hist))
        comm_d, _, _ = _normalize(_to_real(comm_delay))
        sens_f, _, _ = _normalize(_to_real(sens_hist))
        sens_d, _, _ = _normalize(_to_real(sens_delay))

        return {
            "comm_freq": torch.from_numpy(comm_f),    # (2, K, P)
            "comm_delay": torch.from_numpy(comm_d),   # (2, K, P)
            "sens_freq": torch.from_numpy(sens_f),    # (2, K, P)
            "sens_delay": torch.from_numpy(sens_d),   # (2, K, P)
            "target": torch.from_numpy(_to_real(target)),  # (2, K, Q)
            "mean": torch.tensor(mean_c),
            "std": torch.tensor(std_c),
        }
