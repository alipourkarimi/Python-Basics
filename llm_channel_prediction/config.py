"""System and experiment configuration.

All numbers follow Sec. IV of:
    J. He, Z. Ren, J. Yao, H. Hu, T. X. Han, and J. Xu,
    "Sensing-Assisted Channel Prediction in Complex Wireless Environments:
    An LLM-Based Approach," arXiv:2505.09141v1, May 2025.

System setup (paper, Sec. IV):
    * carrier frequency fc = 28 GHz, bandwidth 5.76 MHz,
      K = 48 subcarriers at 60 kHz spacing,
    * BS UPA with Nv = Nh = 4 (N = 16 antennas), half-wavelength spacing,
      single-antenna mobile user (MU),
    * predict Q = 5 future slots from P = 10 historical slots,
    * MU speed uniformly distributed in [10, 100] km/h,
    * 6000 training samples / 600 test samples.

Scatterer counts follow the paper's ray-tracing observation (Fig. 2(b)):
16 dominant communication paths, 26 dominant sensing paths, 14 of which
overlap (shared scatterers)  =>  N0 = 14 shared, N1 = 2 communication-only,
N2 = 12 sensing-only.
"""

from dataclasses import dataclass, field


C_LIGHT = 299_792_458.0  # speed of light [m/s]


@dataclass
class SystemConfig:
    # --- OFDM / array parameters (Sec. IV of the paper) ---
    fc: float = 28e9              # carrier frequency [Hz]
    subcarrier_spacing: float = 60e3   # [Hz]
    n_subcarriers: int = 48       # K
    n_v: int = 4                  # vertical antennas Nv
    n_h: int = 4                  # horizontal antennas Nh
    # CSI sampling interval between consecutive "slots".  The letter does
    # not state it; 100 us is chosen so that at 28 GHz the channel stays
    # strongly correlated slot-to-slot at 10 km/h and substantially
    # decorrelates at 100 km/h, reproducing the difficulty regime of the
    # paper's Fig. 4 (NMSE increasing with MU speed).
    slot_duration: float = 1e-4        # [s]

    # --- prediction task ---
    n_hist: int = 10              # P historical slots
    n_pred: int = 5               # Q future slots

    # --- scatterer geometry (paper Fig. 2(b): 14 shared / 2 comm / 12 sens) ---
    n_shared: int = 14            # N0
    n_comm_only: int = 2          # N1
    n_sens_only: int = 12         # N2

    # --- multipath statistics ---
    tau_max: float = 1.5e-6       # max path delay [s] (~450 m propagation)
    pdp_decay: float = 0.4e-6     # exponential power-delay-profile constant [s]
    v_scatter_max_kmh: float = 30.0    # max speed of scatterer vehicles [km/h]
    elevation_range: float = 60.0      # elevation AoD ~ U(-60, 60) deg
    azimuth_range: float = 90.0       # azimuth AoD  ~ U(-90, 90) deg

    # --- dataset (Sec. IV) ---
    n_train: int = 6000
    n_test: int = 600
    speed_min_kmh: float = 10.0
    speed_max_kmh: float = 100.0

    @property
    def n_antennas(self) -> int:
        return self.n_v * self.n_h

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.fc

    @property
    def n_total_slots(self) -> int:
        return self.n_hist + self.n_pred


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32          # samples per batch (antennas are folded in)
    lr: float = 1e-3
    lr_llm: float = 1e-4          # lower LR for the fine-tuned LLM parts
    weight_decay: float = 1e-5
    # noise-augmented training for the Fig. 5 experiment:
    # "during the training phase, the SNR is set to be uniformly
    #  distributed between 0 and 25 dB for all baselines"
    train_snr_range_db: tuple = (0.0, 25.0)
    seed: int = 2025
    max_antennas: int = 0         # 0 = use all N antennas per sample
    num_workers: int = 0
    device: str = "auto"


@dataclass
class ModelConfig:
    gpt2_name: str = "gpt2"       # backbone (paper Sec. III-C uses GPT-2)
    gpt2_layers: int = 6          # number of transformer decoder blocks kept
    conv_lstm_hidden: int = 32
    n_channel_attention: int = 2  # cascade depth N1 = N2 (paper Fig. 3)
    cross_attention_heads: int = 4
    lstm_hidden: int = 256        # LSTM baseline
    transformer_dim: int = 256    # Transformer baseline
    transformer_layers: int = 2
    cnn_channels: int = 32        # CNN baseline


# Experiment sweeps used for the paper's figures.
@dataclass
class ExperimentConfig:
    fig4_speeds_kmh: tuple = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    fig5_snrs_db: tuple = (0, 5, 10, 15, 20, 25)
    n_test_per_point: int = 200   # test samples generated per sweep point
    results_dir: str = "results"


def quick_preset(sys_cfg: SystemConfig, train_cfg: TrainConfig,
                 model_cfg: ModelConfig, exp_cfg: ExperimentConfig):
    """Shrink everything so the full pipeline runs on a laptop CPU in
    tens of minutes.  Use ``--preset paper`` for the full-scale setup."""
    sys_cfg.n_train = 300
    sys_cfg.n_test = 60
    train_cfg.epochs = 4
    train_cfg.batch_size = 8
    train_cfg.max_antennas = 4
    model_cfg.gpt2_layers = 3
    exp_cfg.fig4_speeds_kmh = (10, 40, 70, 100)
    exp_cfg.fig5_snrs_db = (0, 10, 20, 25)
    exp_cfg.n_test_per_point = 40
    return sys_cfg, train_cfg, model_cfg, exp_cfg
