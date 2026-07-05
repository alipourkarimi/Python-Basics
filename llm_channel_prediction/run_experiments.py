"""End-to-end reproduction of the paper's numerical results.

Runs, in order:
  1. dataset generation (Sec. IV setup),
  2. training of the proposed model, its ablations (Table I), and the
     four baselines (no-sensing / LSTM / Transformer / CNN),
  3. Fig. 4 -- NMSE versus MU speed,
  4. Fig. 5 -- NMSE versus SNR with noisy historical CSI (models
     re-trained with SNR ~ U(0, 25) dB noise augmentation, as in the
     paper),
  5. Table I -- ablation NMSE over all testing velocities.

Usage:
    python run_experiments.py --preset quick   # small-scale CPU demo
    python run_experiments.py --preset paper   # full-scale (GPU advised)

Optional flags: --epochs, --n-train, --skip-fig5, --results-dir, --device.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from channel_dataset import CsiDataset, generate_dataset
from config import (ExperimentConfig, ModelConfig, SystemConfig, TrainConfig,
                    quick_preset)
from models import build_model
from plotting import (plot_nmse_sweep, save_ablation_table,
                      save_results_json)
from train import evaluate_nmse, resolve_device, train_model

FIG4_MODELS = ["proposed", "no_sensing", "lstm", "transformer", "cnn"]
ABLATION_MODELS = ["proposed", "no_sensing", "wo_channel_attention",
                   "wo_cross_attention", "wo_llm"]
FIG5_MODELS = ["proposed", "no_sensing", "lstm", "transformer", "cnn"]


def make_loader(comm, sens, sys_cfg, train_cfg, snr_db=None, shuffle=False,
                seed=0):
    ds = CsiDataset(comm, sens, sys_cfg, snr_db=snr_db,
                    max_antennas=train_cfg.max_antennas, seed=seed)
    return DataLoader(ds, batch_size=train_cfg.batch_size
                      * max(1, train_cfg.max_antennas or sys_cfg.n_antennas),
                      shuffle=shuffle, num_workers=train_cfg.num_workers)


def train_all(model_names, train_loader, val_loader, sys_cfg, model_cfg,
              train_cfg, device, tag_suffix=""):
    trained = {}
    for name in model_names:
        if name in trained:
            continue
        torch.manual_seed(train_cfg.seed)
        model = build_model(name, sys_cfg, model_cfg)
        n_trainable = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
        print(f"\n=== training '{name}{tag_suffix}' "
              f"({n_trainable / 1e6:.2f} M trainable params) ===")
        t0 = time.time()
        trained[name] = train_model(
            model, train_loader, val_loader, train_cfg.epochs,
            train_cfg.lr, train_cfg.lr_llm, train_cfg.weight_decay,
            device, tag=name + tag_suffix)
        print(f"=== '{name}{tag_suffix}' trained in "
              f"{time.time() - t0:.1f} s ===")
    return trained


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=["quick", "paper"], default="quick")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument("--skip-fig5", action="store_true",
                    help="skip the noise-robustness experiment (Fig. 5), "
                         "which requires a second training pass")
    args = ap.parse_args()

    sys_cfg, train_cfg = SystemConfig(), TrainConfig()
    model_cfg, exp_cfg = ModelConfig(), ExperimentConfig()
    if args.preset == "quick":
        quick_preset(sys_cfg, train_cfg, model_cfg, exp_cfg)
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.n_train is not None:
        sys_cfg.n_train = args.n_train
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.results_dir is not None:
        exp_cfg.results_dir = args.results_dir
    train_cfg.device = args.device

    device = resolve_device(train_cfg.device)
    os.makedirs(exp_cfg.results_dir, exist_ok=True)
    print(f"device: {device}   preset: {args.preset}")
    print(f"train/test samples: {sys_cfg.n_train}/{sys_cfg.n_test}, "
          f"epochs: {train_cfg.epochs}, "
          f"antennas per sample: "
          f"{train_cfg.max_antennas or sys_cfg.n_antennas}")

    # ------------------------------------------------------------------ #
    # 1. datasets                                                        #
    # ------------------------------------------------------------------ #
    print("\ngenerating training dataset ...")
    comm_tr, sens_tr, _ = generate_dataset(sys_cfg, sys_cfg.n_train,
                                           seed=train_cfg.seed)
    print("generating test dataset (mixed speeds) ...")
    comm_te, sens_te, _ = generate_dataset(sys_cfg, sys_cfg.n_test,
                                           seed=train_cfg.seed + 1)
    train_loader = make_loader(comm_tr, sens_tr, sys_cfg, train_cfg,
                               shuffle=True, seed=1)
    test_loader = make_loader(comm_te, sens_te, sys_cfg, train_cfg, seed=2)

    # ------------------------------------------------------------------ #
    # 2. train proposed + ablations + baselines (clean CSI)              #
    # ------------------------------------------------------------------ #
    all_names = sorted(set(FIG4_MODELS + ABLATION_MODELS))
    trained = train_all(all_names, train_loader, test_loader,
                        sys_cfg, model_cfg, train_cfg, device)

    results = {"preset": args.preset}

    # ------------------------------------------------------------------ #
    # 3. Fig. 4: NMSE versus MU speed                                    #
    # ------------------------------------------------------------------ #
    print("\n--- Fig. 4: NMSE vs MU speed ---")
    fig4 = {name: [] for name in FIG4_MODELS}
    for v in exp_cfg.fig4_speeds_kmh:
        comm_v, sens_v, _ = generate_dataset(
            sys_cfg, exp_cfg.n_test_per_point,
            seed=train_cfg.seed + 100 + int(v), speed_kmh=float(v))
        loader_v = make_loader(comm_v, sens_v, sys_cfg, train_cfg, seed=3)
        for name in FIG4_MODELS:
            nmse = evaluate_nmse(trained[name], loader_v, device)
            fig4[name].append(nmse)
            print(f"  speed {v:3d} km/h  {name:12s}  NMSE {nmse:.4f}")
    results["fig4"] = {"speeds_kmh": list(exp_cfg.fig4_speeds_kmh), **fig4}
    plot_nmse_sweep(fig4, exp_cfg.fig4_speeds_kmh,
                    "MU speed (km/h)", "NMSE vs MU speed (cf. Fig. 4)",
                    os.path.join(exp_cfg.results_dir, "fig4_nmse_vs_speed.png"))

    # ------------------------------------------------------------------ #
    # 4. Table I: ablation over all testing velocities                   #
    # ------------------------------------------------------------------ #
    print("\n--- Table I: ablation study ---")
    ablation = {}
    for name in ABLATION_MODELS:
        ablation[name] = evaluate_nmse(trained[name], test_loader, device)
        print(f"  {name:22s} NMSE {ablation[name]:.4f}")
    results["ablation"] = ablation
    save_ablation_table(ablation, os.path.join(exp_cfg.results_dir,
                                               "table1_ablation.md"))

    # ------------------------------------------------------------------ #
    # 5. Fig. 5: NMSE versus SNR with noisy historical CSI               #
    # ------------------------------------------------------------------ #
    if not args.skip_fig5:
        print("\n--- Fig. 5: NMSE vs SNR (noise-augmented training) ---")
        noisy_train_loader = make_loader(
            comm_tr, sens_tr, sys_cfg, train_cfg,
            snr_db=train_cfg.train_snr_range_db, shuffle=True, seed=4)
        trained_noisy = train_all(FIG5_MODELS, noisy_train_loader,
                                  test_loader, sys_cfg, model_cfg,
                                  train_cfg, device, tag_suffix="+noise")
        fig5 = {name: [] for name in FIG5_MODELS}
        for snr in exp_cfg.fig5_snrs_db:
            loader_s = make_loader(comm_te, sens_te, sys_cfg, train_cfg,
                                   snr_db=float(snr), seed=5)
            for name in FIG5_MODELS:
                nmse = evaluate_nmse(trained_noisy[name], loader_s, device)
                fig5[name].append(nmse)
                print(f"  SNR {snr:3d} dB  {name:12s}  NMSE {nmse:.4f}")
        results["fig5"] = {"snrs_db": list(exp_cfg.fig5_snrs_db), **fig5}
        plot_nmse_sweep(fig5, exp_cfg.fig5_snrs_db,
                        "SNR (dB)", "NMSE vs SNR, noisy historical CSI "
                        "(cf. Fig. 5)",
                        os.path.join(exp_cfg.results_dir,
                                     "fig5_nmse_vs_snr.png"))

    save_results_json(results, os.path.join(exp_cfg.results_dir,
                                            "results.json"))
    print("\nall experiments finished.")


if __name__ == "__main__":
    main()
