# Reproduction: Sensing-Assisted Channel Prediction with an LLM

Complete simulation code reproducing the numerical results of

> J. He, Z. Ren, J. Yao, H. Hu, T. X. Han, and J. Xu,
> **"Sensing-Assisted Channel Prediction in Complex Wireless Environments:
> An LLM-Based Approach,"** arXiv:2505.09141v1, May 2025.

An ISAC base station with a 4×4 UPA predicts the next **Q = 5** slots of its
bi-static communication CSI from the previous **P = 10** slots of both
communication CSI and mono-static sensing CSI, exploiting the multipath
components contributed by *shared scatterers* in both channels.

## What is reproduced

| Paper item | This code |
|---|---|
| System model, Eqs. (1)–(2) | `channel_dataset.py` — geometric multipath ISAC channels (UPA steering vectors, Doppler, delays, shared/comm/sensing scatterers) |
| Preprocessor (Sec. III-A) | `channel_dataset.py::CsiDataset` — K-point IDFT to the delay domain, complex→real, per-antenna parallel processing, normalization |
| Channel attention + cross-attention fusion, Eq. (4) (Sec. III-B) | `models.py::ChannelAttention` (cascaded ConvLSTM cells), `models.py::CrossAttentionFusion` |
| GPT-2 backbone with frozen attention/FFN, fine-tuned LayerNorm + positional embedding, Eqs. (5)–(7) (Sec. III-C/D) | `models.py::GPT2Backbone`, `SensingAssistedLLMPredictor` |
| NMSE loss, Eq. (8) | `train.py::nmse_loss` |
| Baselines: no sensing / LSTM / Transformer / CNN (Sec. IV) | `models.py` |
| Fig. 4 — NMSE vs MU speed | `run_experiments.py` → `results/fig4_nmse_vs_speed.png` |
| Fig. 5 — NMSE vs SNR with noisy historical CSI (training SNR ~ U(0, 25) dB) | `run_experiments.py` → `results/fig5_nmse_vs_snr.png` |
| Table I — ablation (w/o sensing, w/o channel attention, w/o cross attention, w/o LLM) | `run_experiments.py` → `results/table1_ablation.md` |

System parameters follow Sec. IV exactly: 28 GHz carrier, 5.76 MHz bandwidth,
K = 48 subcarriers at 60 kHz spacing, N<sub>v</sub> = N<sub>h</sub> = 4,
MU speed uniform in [10, 100] km/h, 6000 training / 600 test samples
(`--preset paper`).

## Install & run

```bash
pip install -r requirements.txt

# small-scale demo that runs on a laptop CPU (~30–60 min):
python run_experiments.py --preset quick

# full paper-scale experiment (GPU strongly recommended):
python run_experiments.py --preset paper
```

Useful flags: `--epochs N`, `--n-train N`, `--batch-size B`,
`--device cuda|cpu`, `--skip-fig5` (skips the second, noise-augmented
training pass), `--results-dir DIR`.

Outputs land in `results/`: the two figures (PNG), the ablation table
(Markdown), and all raw numbers (`results.json`).

The first run downloads the pre-trained GPT-2 weights from Hugging Face; if
the machine is offline the code falls back to a randomly initialized GPT-2
of the same architecture (a warning is printed — results will be weaker
because the pre-trained knowledge that the paper's approach relies on is
missing).

## Faithfulness notes / deviations

1. **Dataset**: the paper generates CSI with NVIDIA Sionna ray tracing in a
   street scene whose geometry is not published. This code instead draws the
   per-path parameters at random with the statistics the paper reports
   (14 shared / 2 communication-only / 12 sensing-only dominant paths from
   Fig. 2(b); quasi-stationary angles/delays/Dopplers over the P+Q slots;
   slot-to-slot variation from the phase rotation of many paths) and then
   evaluates the channel equations (1)–(2) exactly. Absolute NMSE values
   therefore differ from the paper's, but the *relative ordering* of the
   schemes — the actual claim of the paper — is reproduced.
2. **Eq. (4)** as printed applies `CA(·)` to the *sum* of the two feature
   streams, while the text says cross-attention is used "instead of simply
   adding them". We implement the standard reading: communication features
   are queries, sensing features are keys/values, with a residual
   connection; the "w/o cross attention" ablation is the plain addition.
3. Hyper-parameters the letter does not specify (ConvLSTM hidden size,
   cascade depth, number of GPT-2 blocks kept, learning rates, epochs) are
   set to values typical of the cited LLM4CP line of work; all are in
   `config.py`.
4. The letter does not state the slot (CSI sampling) interval. It is set to
   100 µs, so that at 28 GHz the channel stays strongly correlated
   slot-to-slot at 10 km/h and substantially decorrelates at 100 km/h —
   reproducing the difficulty regime of the paper's Fig. 4 (NMSE increasing
   with MU speed).

## Files

```
config.py            all system / model / training / experiment parameters
channel_dataset.py   ISAC channel generator (Eqs. (1)-(2)) + preprocessor
models.py            proposed LLM predictor, ablations, and baselines
train.py             NMSE loss (Eq. (8)), training and evaluation loops
plotting.py          Fig. 4 / Fig. 5 plots and Table I writer
run_experiments.py   end-to-end experiment driver
```
