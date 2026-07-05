# Large Language Models (LLMs) for ISAC

LLMs enter ISAC research in three roles: as **optimizers** (steering search/evolutionary
algorithms with in-context reasoning), as **multimodal perceivers** (fusing CSI, radar
returns, and vision for beam prediction or target recognition), and as **network agents**
(autonomous management of 6G resources). Foundation models trained directly on wireless
channel data ("wireless GPTs") are an emerging fourth direction.

---

## 1. Large Language Model Based Multi-Objective Optimization for Integrated Sensing and Communications in UAV Networks

- **arXiv:** [2410.05062](https://arxiv.org/abs/2410.05062) · Oct 2024 · eess.SY / cs.AI

**Brief.** Considers a UAV that simultaneously communicates with ground users and senses
targets (UAV-enabled ISAC). The joint design of UAV trajectory/deployment and transmit
beamforming is a multi-objective problem — communication rate vs. sensing performance
trade-off. The authors propose **LEDMA**, an LLM-enabled decomposition-based
multi-objective evolutionary algorithm: the problem is decomposed into scalar
sub-problems (MOEA/D style) and *LLMs act as the search operators*, driven by carefully
engineered prompts that inject the optimization context in-context. Results show LEDMA
outperforms standard multi-objective baselines and finds better Pareto fronts.

**Why it matters:** one of the first concrete demonstrations that off-the-shelf LLMs can
replace hand-designed crossover/mutation operators in ISAC network optimization.

---

## 2. WirelessGPT: A Generative Pre-trained Multi-task Learning Framework for Wireless Communication

- **arXiv:** [2502.06877](https://arxiv.org/abs/2502.06877) · Feb 2025 · eess.SP

**Brief.** Proposes a **wireless foundation model** pre-trained on large-scale wireless
channel data to learn a universal channel representation, then fine-tuned for multiple
downstream tasks — channel estimation, channel prediction, and **sensing tasks (e.g.,
human activity recognition)** within one model. This is one of the early attempts to
unify communication and sensing tasks inside a single generative pre-trained model,
which is exactly the ISAC vision at the model level: one representation serving both
functions. Reported gains over task-specific baselines grow with model size, suggesting
scaling laws carry over to wireless data.

**Why it matters:** points toward "one foundation model per network" instead of dozens of
task-specific neural networks — a natural AI counterpart to ISAC's hardware unification.

---

## 3. When Large Language Model Agents Meet 6G Networks: Perception, Grounding, and Alignment

- **arXiv:** [2401.07764](https://arxiv.org/abs/2401.07764) · Jan 2024 · cs.NI

**Brief.** A vision/framework paper on running **LLM agents split between mobile devices
and 6G edge servers** (small local model + large edge model). It maps the classic agent
loop — perception, grounding, alignment — onto 6G capabilities and argues that **ISAC is
the perception layer of LLM agents**: integrated sensing supplies the environmental
information (location, motion, surroundings) that agents need to ground their reasoning
in the physical world, while the communication half moves model context between device
and edge efficiently.

**Why it matters:** frames ISAC not just as a radio technique but as the "eyes and ears"
of future network-native AI agents.

---

## 4. Chain-of-Thought for Large Language Model-empowered Wireless Communications

- **arXiv:** [2505.22320](https://arxiv.org/abs/2505.22320) · May 2025 · eess.SP

**Brief.** Studies how **chain-of-thought (CoT) prompting** improves LLM performance on
wireless-domain problems, where naive prompting often fails because the tasks require
multi-step numerical and physical reasoning. The paper walks through wireless use cases
(including sensing/ISAC-flavored tasks) and shows that structured intermediate reasoning
steps substantially improve solution quality, positioning CoT as a cheap alternative to
fine-tuning for wireless problem solving.

**Why it matters:** practical guidance on *how* to prompt general-purpose LLMs for
ISAC/wireless engineering tasks rather than training new models.

---

## 5. Graph-Enhanced LLM for SWAN-ISAC

- **arXiv:** [2604.10256](https://arxiv.org/abs/2604.10256) · Apr 2026 · eess.SP

**Brief.** (2026 paper — brief based on search-result metadata.) Combines a **CSI-induced
self-graph neural network** with an **LLM backbone fine-tuned via LoRA** (low-rank
adaptation) to capture the interactions among communication users and sensing targets in
a space-water-air-network (SWAN) ISAC setting. The graph structure encodes who interferes
with / senses whom, and the LLM provides the sequence-modeling capacity, with LoRA keeping
the fine-tuning cost low.

**Why it matters:** shows the current frontier — hybrid GNN + LLM architectures applied
to multi-domain (space/air/water) ISAC networks.

---

### Category takeaway

LLM-for-ISAC work is young but moving fast: 2024 papers used LLMs *around* the problem
(as optimizers or agents); 2025–2026 papers increasingly put wireless data *inside* the
model (foundation models, LoRA fine-tuning on CSI). The open challenges repeatedly named
are hallucination in numeric tasks, latency of large models at the network edge, and the
lack of large public ISAC datasets for pre-training.
