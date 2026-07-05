"""Network architectures.

* ``SensingAssistedLLMPredictor`` -- the proposed approach of Sec. III
  (Fig. 3): preprocessed communication + sensing CSI -> ConvLSTM channel
  attention (Eq. (4)) -> cross-attention fusion -> FC lift to the LLM
  feature dimension -> non-learnable positional encoding (Eq. (5)) ->
  frozen GPT-2 backbone (Eq. (6)) -> FC output head (Eq. (7)).
  Constructor flags implement the paper's ablations (Table I) and the
  "no sensing assistance" scheme.

* ``LSTMPredictor`` / ``TransformerPredictor`` / ``CNNPredictor`` -- the
  deep-learning baselines of Sec. IV, which use historical communication
  CSI only and process antennas in parallel, with NMSE training loss.

All models take normalized real tensors of shape (B, 2, K, P) and output
the normalized prediction (B, 2, K, Q); de-normalization (Eq. (7)) is done
by the caller with the statistics returned by the dataset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import ModelConfig, SystemConfig


# --------------------------------------------------------------------------
# ConvLSTM channel attention (Sec. III-B)
# --------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """1-D convolutional LSTM cell (convolution along the delay/subcarrier
    axis K), following the ConvLSTM design of [5] referenced by the paper."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv1d(in_ch + hidden_ch, 4 * hidden_ch, kernel,
                              padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, k, device):
        z = torch.zeros(batch, self.hidden_ch, k, device=device)
        return z, z.clone()


class ChannelAttention(nn.Module):
    """CL^(N)(X_tau, X) of Eq. (4): N cascaded ConvLSTM cells that extract
    joint spatiotemporal features from the delay-domain and frequency-domain
    CSI of one branch (communication or sensing).

    Input:  two tensors (B, 2, K, P);  output: (B, P, 2K) feature sequence."""

    def __init__(self, n_cascade: int, hidden_ch: int, k: int):
        super().__init__()
        self.cells = nn.ModuleList(
            [ConvLSTMCell(4 if i == 0 else hidden_ch, hidden_ch)
             for i in range(n_cascade)])
        self.proj = nn.Conv1d(hidden_ch, 2, kernel_size=1)
        self.k = k

    def forward(self, x_freq, x_delay):
        x = torch.cat([x_freq, x_delay], dim=1)      # (B, 4, K, P)
        b, _, k, p = x.shape
        states = [cell.init_state(b, k, x.device) for cell in self.cells]
        outputs = []
        for t in range(p):
            inp = x[..., t]                           # (B, 4, K)
            for i, cell in enumerate(self.cells):
                h, c = cell(inp, states[i])
                states[i] = (h, c)
                inp = h
            outputs.append(self.proj(inp))            # (B, 2, K)
        out = torch.stack(outputs, dim=1)              # (B, P, 2, K)
        return out.reshape(b, p, 2 * k)                # (B, P, 2K)


class CrossAttentionFusion(nn.Module):
    """CA(.) of Eq. (4): fuses the communication and sensing feature
    sequences with multi-head cross-attention (communication features as
    queries, sensing features as keys/values) plus a residual connection,
    'instead of simply adding them' (Sec. III-B)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, comm_feat, sens_feat):
        fused, _ = self.attn(comm_feat, sens_feat, sens_feat)
        return self.norm(comm_feat + fused)


def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    """Non-learnable positional encoding X_PE of Eq. (5)."""
    pe = torch.zeros(length, dim)
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# --------------------------------------------------------------------------
# GPT-2 backbone (Sec. III-C)
# --------------------------------------------------------------------------

class GPT2Backbone(nn.Module):
    """Pre-trained GPT-2 backbone, Eq. (6).

    Per Sec. III-D, the multi-head attention and feed-forward layers are
    FROZEN, while the layer-normalization and positional-embedding layers
    stay trainable to adapt the LLM to the channel-prediction task.
    Only the first ``n_layers`` transformer blocks are kept.

    If the pre-trained weights cannot be downloaded (offline machine) the
    backbone falls back to a randomly initialized GPT-2 of the same shape
    and prints a warning: the code still runs, but the benefit of
    pre-trained knowledge is lost.
    """

    def __init__(self, name: str = "gpt2", n_layers: int = 6):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        try:
            gpt2 = GPT2Model.from_pretrained(name)
            print(f"[GPT2Backbone] loaded pre-trained '{name}'")
        except Exception as err:  # offline fallback
            print(f"[GPT2Backbone] WARNING: could not download '{name}' "
                  f"({err}); using a randomly initialized GPT-2.")
            gpt2 = GPT2Model(GPT2Config())
        gpt2.h = gpt2.h[:n_layers]
        gpt2.config.n_layer = n_layers
        self.gpt2 = gpt2
        self.feature_dim = gpt2.config.n_embd

        for name_, param in self.gpt2.named_parameters():
            # freeze attention + MLP; fine-tune LayerNorm + positional emb.
            if "ln" in name_ or "wpe" in name_:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        # x: (B, P, F) fed as input embeddings ("CSI tokens")
        return self.gpt2(inputs_embeds=x).last_hidden_state


# --------------------------------------------------------------------------
# the proposed network (Fig. 3)
# --------------------------------------------------------------------------

class SensingAssistedLLMPredictor(nn.Module):
    """Proposed LLM-based sensing-assisted channel predictor.

    Ablation flags (Table I):
        use_sensing=False            -> "W/o sensing"
        use_channel_attention=False  -> "W/o channel attention"
        use_cross_attention=False    -> "W/o cross attention" (simple add)
        use_llm=False                -> "W/o LLM" (backbone removed)
    """

    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig,
                 use_sensing=True, use_channel_attention=True,
                 use_cross_attention=True, use_llm=True):
        super().__init__()
        k, p, q = sys_cfg.n_subcarriers, sys_cfg.n_hist, sys_cfg.n_pred
        self.use_sensing = use_sensing
        self.use_channel_attention = use_channel_attention
        self.use_cross_attention = use_cross_attention and use_sensing
        self.use_llm = use_llm
        feat = 2 * k

        if use_channel_attention:
            self.ca_comm = ChannelAttention(model_cfg.n_channel_attention,
                                            model_cfg.conv_lstm_hidden, k)
            if use_sensing:
                self.ca_sens = ChannelAttention(
                    model_cfg.n_channel_attention,
                    model_cfg.conv_lstm_hidden, k)
        else:
            # ablation: plain per-slot linear feature extraction
            self.ca_comm = nn.Linear(2 * feat, feat)
            if use_sensing:
                self.ca_sens = nn.Linear(2 * feat, feat)

        if self.use_cross_attention:
            self.fusion = CrossAttentionFusion(
                feat, model_cfg.cross_attention_heads)

        if use_llm:
            self.backbone = GPT2Backbone(model_cfg.gpt2_name,
                                         model_cfg.gpt2_layers)
            f = self.backbone.feature_dim
        else:
            self.backbone = None
            f = feat  # skip the lift; features go straight to the head

        self.lift = nn.Linear(feat, f)
        self.register_buffer("pos_enc",
                             sinusoidal_positional_encoding(p, f))
        self.head = nn.Linear(p * f, q * feat)
        self.k, self.p, self.q = k, p, q

    def _branch_features(self, module, x_freq, x_delay):
        if self.use_channel_attention:
            return module(x_freq, x_delay)             # (B, P, 2K)
        b, _, k, p = x_freq.shape
        x = torch.cat([x_freq, x_delay], dim=1)        # (B, 4, K, P)
        x = x.permute(0, 3, 1, 2).reshape(b, p, 4 * k)
        return module(x)                                # (B, P, 2K)

    def forward(self, comm_freq, comm_delay, sens_freq=None, sens_delay=None):
        comm_feat = self._branch_features(self.ca_comm, comm_freq, comm_delay)
        if self.use_sensing:
            sens_feat = self._branch_features(self.ca_sens,
                                              sens_freq, sens_delay)
            if self.use_cross_attention:
                fused = self.fusion(comm_feat, sens_feat)   # Eq. (4)
            else:
                fused = comm_feat + sens_feat               # ablation: add
        else:
            fused = comm_feat

        x = self.lift(fused)                                # (B, P, F)
        x = x + self.pos_enc.unsqueeze(0)                   # Eq. (5)
        if self.use_llm:
            x = self.backbone(x)                            # Eq. (6)
        b = x.shape[0]
        out = self.head(x.reshape(b, -1))                   # Eq. (7)
        return out.reshape(b, 2, self.k, self.q)

    def parameter_groups(self, lr: float, lr_llm: float):
        """Separate LR for the fine-tuned LLM parts (LN + wpe)."""
        llm_params, other = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (llm_params if name.startswith("backbone") else other).append(
                param)
        groups = [{"params": other, "lr": lr}]
        if llm_params:
            groups.append({"params": llm_params, "lr": lr_llm})
        return groups


# --------------------------------------------------------------------------
# baselines (Sec. IV) -- historical communication CSI only
# --------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """LSTM baseline: memory cells with multiplicative gates for long-term
    dependency; per-antenna parallel processing, NMSE loss."""

    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig):
        super().__init__()
        k, p, q = sys_cfg.n_subcarriers, sys_cfg.n_hist, sys_cfg.n_pred
        h = model_cfg.lstm_hidden
        self.lstm = nn.LSTM(2 * k, h, num_layers=2, batch_first=True)
        self.head = nn.Linear(p * h, q * 2 * k)
        self.k, self.q = k, q

    def forward(self, comm_freq, comm_delay=None, *_, **__):
        b, _, k, p = comm_freq.shape
        x = comm_freq.permute(0, 3, 1, 2).reshape(b, p, 2 * k)
        out, _ = self.lstm(x)
        out = self.head(out.reshape(b, -1))
        return out.reshape(b, 2, self.k, self.q)


class TransformerPredictor(nn.Module):
    """Transformer baseline [4]: parallel attention-based predictor."""

    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig):
        super().__init__()
        k, p, q = sys_cfg.n_subcarriers, sys_cfg.n_hist, sys_cfg.n_pred
        d = model_cfg.transformer_dim
        self.embed = nn.Linear(2 * k, d)
        self.register_buffer("pos_enc",
                             sinusoidal_positional_encoding(p, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=4 * d,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=model_cfg.transformer_layers)
        self.head = nn.Linear(p * d, q * 2 * k)
        self.k, self.q = k, q

    def forward(self, comm_freq, comm_delay=None, *_, **__):
        b, _, k, p = comm_freq.shape
        x = comm_freq.permute(0, 3, 1, 2).reshape(b, p, 2 * k)
        x = self.embed(x) + self.pos_enc.unsqueeze(0)
        x = self.encoder(x)
        out = self.head(x.reshape(b, -1))
        return out.reshape(b, 2, self.k, self.q)


class CNNPredictor(nn.Module):
    """CNN baseline: treats the time-frequency CSI (2, K, P) as a 2-D image
    and predicts the (2, K, Q) continuation."""

    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig):
        super().__init__()
        k, p, q = sys_cfg.n_subcarriers, sys_cfg.n_hist, sys_cfg.n_pred
        c = model_cfg.cnn_channels
        self.net = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(c * k * p, q * 2 * k)
        self.k, self.q = k, q

    def forward(self, comm_freq, comm_delay=None, *_, **__):
        b = comm_freq.shape[0]
        x = self.net(comm_freq)
        out = self.head(x.reshape(b, -1))
        return out.reshape(b, 2, self.k, self.q)


# --------------------------------------------------------------------------
# model zoo
# --------------------------------------------------------------------------

def build_model(name: str, sys_cfg: SystemConfig,
                model_cfg: ModelConfig) -> nn.Module:
    if name == "proposed":
        return SensingAssistedLLMPredictor(sys_cfg, model_cfg)
    if name == "no_sensing":
        return SensingAssistedLLMPredictor(sys_cfg, model_cfg,
                                           use_sensing=False)
    if name == "wo_channel_attention":
        return SensingAssistedLLMPredictor(sys_cfg, model_cfg,
                                           use_channel_attention=False)
    if name == "wo_cross_attention":
        return SensingAssistedLLMPredictor(sys_cfg, model_cfg,
                                           use_cross_attention=False)
    if name == "wo_llm":
        return SensingAssistedLLMPredictor(sys_cfg, model_cfg,
                                           use_llm=False)
    if name == "lstm":
        return LSTMPredictor(sys_cfg, model_cfg)
    if name == "transformer":
        return TransformerPredictor(sys_cfg, model_cfg)
    if name == "cnn":
        return CNNPredictor(sys_cfg, model_cfg)
    raise ValueError(f"unknown model '{name}'")
