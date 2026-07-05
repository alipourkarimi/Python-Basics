# AI-Empowered Integrated Sensing and Communication (ISAC) — arXiv Research Collection

A curated collection of arXiv papers at the intersection of **Integrated Sensing and
Communication (ISAC)** and **Artificial Intelligence** — covering Large Language Models
(LLMs), Generative AI / diffusion models, (Deep) Reinforcement Learning, and classical
Deep Learning approaches.

ISAC is a key enabling technology for 6G in which one waveform, one spectrum, and one
hardware platform serve **both** radar-style sensing (detection, localization, tracking)
and data communication. Because the joint design space is huge and hard to optimize with
closed-form methods, AI techniques have become the dominant research direction.

> Compiled on **2026-07-05** from arXiv searches. Each brief is based on the paper's
> abstract and public metadata — always read the original paper before citing.

---

## Repository structure

| File | Contents |
|---|---|
| [`papers/01-llm-for-isac.md`](papers/01-llm-for-isac.md) | Large Language Models & foundation models for ISAC |
| [`papers/02-generative-ai-for-isac.md`](papers/02-generative-ai-for-isac.md) | Generative AI & diffusion models for ISAC |
| [`papers/03-reinforcement-learning-for-isac.md`](papers/03-reinforcement-learning-for-isac.md) | (Deep) Reinforcement Learning for ISAC |
| [`papers/04-deep-learning-for-isac.md`](papers/04-deep-learning-for-isac.md) | Deep learning, surveys & tutorials for ISAC |
| [`scripts/fetch_arxiv_isac.py`](scripts/fetch_arxiv_isac.py) | Python script that re-runs the arXiv search so the collection can be refreshed |

## Paper index

| # | arXiv ID | Title (short) | AI technique |
|---|---|---|---|
| 1 | [2410.05062](https://arxiv.org/abs/2410.05062) | LLM-based multi-objective optimization for ISAC UAV networks | LLM + evolutionary optimization |
| 2 | [2502.06877](https://arxiv.org/abs/2502.06877) | WirelessGPT: generative pre-trained multi-task wireless model | Wireless foundation model |
| 3 | [2401.07764](https://arxiv.org/abs/2401.07764) | LLM agents meet 6G networks | LLM agents / edge AI |
| 4 | [2505.22320](https://arxiv.org/abs/2505.22320) | Chain-of-Thought for LLM-empowered wireless | LLM reasoning (CoT) |
| 5 | [2604.10256](https://arxiv.org/abs/2604.10256) | Graph-enhanced LLM for SWAN-ISAC | GNN + LLM (LoRA) |
| 6 | [2502.12622](https://arxiv.org/abs/2502.12622) | GenAI data augmentation for wireless sensing in ISAC | Conditional diffusion |
| 7 | [2408.11398](https://arxiv.org/abs/2408.11398) | GenAI-based secure wireless sensing (DFSS) | Diffusion models |
| 8 | [2506.06942](https://arxiv.org/abs/2506.06942) | Conditional denoising diffusion for cell-free 6G channel estimation | Diffusion + transformer |
| 9 | [2406.00408](https://arxiv.org/abs/2406.00408) | Optimizing 6G ISAC via expert networks | Mixture-of-experts + GenAI |
| 10 | [2510.25496](https://arxiv.org/abs/2510.25496) | Dynamic beamforming & power allocation via DRL | Deep RL |
| 11 | [2506.18560](https://arxiv.org/abs/2506.18560) | Digital-twin-assisted DRL beam selection (cell-free massive MIMO) | Deep RL + digital twin |
| 12 | [2508.19566](https://arxiv.org/abs/2508.19566) | Energy-efficient learning-based beamforming for ISAC V2X | Deep RL |
| 13 | [2405.20168](https://arxiv.org/abs/2405.20168) | Aerial RIS-assisted ISAC with DRL | Deep RL |
| 14 | [2412.01077](https://arxiv.org/abs/2412.01077) | Memory-based RL for ISAC | Recurrent / memory RL |
| 15 | [2509.06968](https://arxiv.org/abs/2509.06968) | DL techniques for ISAC: state of the art (survey) | Survey |
| 16 | [2504.13363](https://arxiv.org/abs/2504.13363) | AI-empowered ISAC (tutorial) | Tutorial (SL/UL/RL) |
| 17 | [2307.15074](https://arxiv.org/abs/2307.15074) | ISAC-NET: model-driven DL for passive sensing + communication | Model-driven DL |
| 18 | [2403.01480](https://arxiv.org/abs/2403.01480) | DL-based design of uplink ISAC | Deep learning |
| 19 | [2402.09439](https://arxiv.org/abs/2402.09439) | DL channel estimation for IRS-assisted ISAC | DNNs |
| 20 | [2509.21118](https://arxiv.org/abs/2509.21118) | Neural ISAC for the MIMO-OFDM downlink | Neural signal processing |
| 21 | [2509.02137](https://arxiv.org/abs/2509.02137) | High-resolution sensing in communication-centric ISAC | DL + parametric methods |

## Key takeaways across the literature

1. **LLMs are moving from "chatbot" to "optimizer/controller"** — recent work uses LLMs
   as search operators inside evolutionary algorithms, as multimodal reasoners over CSI,
   and as network-management agents, rather than only as language interfaces.
2. **Diffusion models are the workhorse of generative ISAC** — used for channel/CSI data
   augmentation, secure sensing, and channel estimation, where their ability to model
   complex high-dimensional distributions beats GAN/VAE baselines.
3. **DRL dominates real-time resource allocation** — beamforming, power allocation, beam
   selection and RIS/trajectory control are naturally sequential decision problems;
   papers consistently report orders-of-magnitude faster inference than convex/SDR
   solvers at a modest performance gap.
4. **Digital twins and model-driven DL narrow the sim-to-real gap** — pure black-box
   learning is giving way to hybrid designs that embed known signal-processing structure
   (ISAC-NET) or train against a synchronized virtual replica of the network.

## Refreshing this collection

```bash
python scripts/fetch_arxiv_isac.py            # print latest results to stdout
python scripts/fetch_arxiv_isac.py -o out.md  # write a fresh markdown report
```

The script queries the official [arXiv API](https://info.arxiv.org/help/api/index.html)
(`export.arxiv.org/api/query`) for each AI-technique category and formats the results as
markdown. arXiv asks automated clients to stay under 1 request / 3 seconds — the script
respects that.
