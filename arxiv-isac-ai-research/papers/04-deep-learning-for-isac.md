# Deep Learning for ISAC — Surveys, Tutorials, and Core Designs

Beyond LLMs, GenAI, and RL, a large body of ISAC work applies "classical" deep learning:
DNN receivers that demodulate data *and* estimate targets, learned channel estimators,
and end-to-end neural transceivers. This file collects the main surveys/tutorials (start
here if you're new) and representative technical designs.

---

## Surveys & tutorials

### 1. Deep Learning-based Techniques for Integrated Sensing and Communication Systems: State-of-the-Art, Challenges, and Opportunities

- **arXiv:** [2509.06968](https://arxiv.org/abs/2509.06968) · Sep 2025 · eess.SP

**Brief.** A comprehensive survey of DL across the ISAC stack. Covers, among others:
fully connected DNN receivers that demodulate communication symbols while simultaneously
processing the received signal for sensing, and ISAC receivers whose neural networks
jointly demodulate data and estimate target **range and velocity under hardware
imperfections** (where model-based methods degrade). Ends with a challenges/opportunities
roadmap: datasets, generalization across environments, and complexity at the edge.

**Recommended as the entry-point survey for this collection.**

### 2. AI-Empowered Integrated Sensing and Communications

- **arXiv:** [2504.13363](https://arxiv.org/abs/2504.13363) · Apr 2025 · eess.SP

**Brief.** A tutorial-style treatment of AI for the two central physical-layer design
problems of ISAC: **waveform design** and **beamforming**. Systematically walks through
supervised, unsupervised, and reinforcement-learning approaches to each, comparing them
against optimization-based designs, and identifying when learning genuinely helps
(non-convexity, imperfect models, real-time constraints) versus when classical methods
remain preferable.

---

## Technical designs

### 3. ISAC-NET: Model-driven Deep Learning for Integrated Passive Sensing and Communication

- **arXiv:** [2307.15074](https://arxiv.org/abs/2307.15074) · Jul 2023 · eess.SP

**Brief.** Instead of a black-box network, ISAC-NET is **model-driven**: it unfolds the
known signal-processing pipeline into learnable blocks — a passive-sensing module, a
signal-detection module, and a channel-reconstruction module — processed block by block.
It outputs **passive sensing results and demodulated communication symbols
simultaneously**, showing better sensing accuracy than traditional passive sensing at
comparable/better demodulation performance.

**Why it matters:** the model-driven (deep-unfolding) philosophy keeps physical
interpretability and needs far less training data than end-to-end black boxes.

### 4. Deep Learning-based Design of Uplink Integrated Sensing and Communication

- **arXiv:** [2403.01480](https://arxiv.org/abs/2403.01480) · Mar 2024 · eess.SP

**Brief.** Addresses the less-studied **uplink** ISAC setting, where the base station
must decode user data while sensing from the same received signals. Proposes DL-based
receiver designs that handle the interference between the two roles, outperforming
conventional separated processing, and analyzes the achievable
communication–sensing operating points.

### 5. Deep-Learning-Based Channel Estimation for IRS-Assisted ISAC System

- **arXiv:** [2402.09439](https://arxiv.org/abs/2402.09439) · Feb 2024 · eess.SP

**Brief.** In an intelligent-reflecting-surface (IRS)-assisted ISAC system, the cascaded
channels (BS→IRS→user/target) are high-dimensional and hard to estimate. The paper
builds a **deep-learning estimation framework with two DNN architectures** tailored to
different propagation environments, estimating the sensing and communication channels
and beating least-squares/traditional baselines, especially at low SNR.

### 6. Neural Integrated Sensing and Communication for the MIMO-OFDM Downlink

- **arXiv:** [2509.21118](https://arxiv.org/abs/2509.21118) · Sep 2025 · eess.SP

**Brief.** Presents a signal-processing framework for **neural ISAC in MIMO-OFDM
downlink** systems — the waveform family used by 5G/6G. Data-driven components are
inserted into the transceiver chain so that one learned design serves both the
communication link and the sensing function over standard OFDM signals, keeping
compatibility with existing air interfaces.

### 7. High-Resolution Sensing in Communication-Centric ISAC: Deep Learning and Parametric Methods

- **arXiv:** [2509.02137](https://arxiv.org/abs/2509.02137) · Sep 2025 · eess.SP

**Brief.** Communication-centric ISAC reuses communication waveforms (like OFDM) for
sensing, which limits resolution. This work compares and combines **parametric
(model-based super-resolution) methods with deep learning** to push sensing resolution
beyond classical FFT-based processing, characterizing when learned methods beat
parametric estimators and proposing hybrid pipelines.

---

### Category takeaway

The DL-for-ISAC literature is converging on **hybrid designs**: embed the known physics
(OFDM structure, array manifolds, estimation pipelines) and let neural networks handle
what models get wrong (hardware impairments, complex clutter, non-convex couplings).
Pure end-to-end black boxes are increasingly rare outside benchmark studies.
