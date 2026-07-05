"""Training and evaluation utilities.

The loss is the NMSE of Eq. (8):

    L = sum_n || H_hat_c,n - H_c,n ||_F^2  /  sum_n || H_c,n ||_F^2 ,

computed on the DE-NORMALIZED prediction against the ground-truth CSI,
summed over all antenna pairs in the batch.  The same metric is used for
validation and testing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import SensingAssistedLLMPredictor


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denorm(pred: torch.Tensor, mean: torch.Tensor,
           std: torch.Tensor) -> torch.Tensor:
    """De-normalization of Eq. (7): map the network output back to the
    physical CSI scale using the per-item statistics."""
    return pred * std.view(-1, 1, 1, 1) + mean.view(-1, 1, 1, 1)


def nmse_terms(pred_denorm: torch.Tensor, target: torch.Tensor):
    err = torch.sum((pred_denorm - target) ** 2)
    ref = torch.sum(target ** 2)
    return err, ref


def nmse_loss(pred_denorm: torch.Tensor, target: torch.Tensor):
    err, ref = nmse_terms(pred_denorm, target)
    return err / (ref + 1e-12)


def _forward(model: nn.Module, batch: dict, device: torch.device):
    kwargs = dict(
        comm_freq=batch["comm_freq"].to(device),
        comm_delay=batch["comm_delay"].to(device),
    )
    if isinstance(model, SensingAssistedLLMPredictor) and model.use_sensing:
        kwargs["sens_freq"] = batch["sens_freq"].to(device)
        kwargs["sens_delay"] = batch["sens_delay"].to(device)
    pred = model(**kwargs)
    return denorm(pred, batch["mean"].to(device), batch["std"].to(device))


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, epochs: int, lr: float,
                lr_llm: float, weight_decay: float,
                device: torch.device, tag: str = "") -> nn.Module:
    model.to(device)
    if isinstance(model, SensingAssistedLLMPredictor):
        groups = model.parameter_groups(lr, lr_llm)
    else:
        groups = [{"params": model.parameters(), "lr": lr}]
    opt = torch.optim.Adam(groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        run_err = run_ref = 0.0
        for batch in train_loader:
            opt.zero_grad()
            pred = _forward(model, batch, device)
            loss = nmse_loss(pred, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], 1.0)
            opt.step()
            with torch.no_grad():
                err, ref = nmse_terms(pred, batch["target"].to(device))
                run_err += err.item()
                run_ref += ref.item()
        sched.step()
        val = evaluate_nmse(model, val_loader, device)
        print(f"[{tag}] epoch {epoch:3d}/{epochs}  "
              f"train NMSE {run_err / run_ref:.4f}  val NMSE {val:.4f}")
    return model


@torch.no_grad()
def evaluate_nmse(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> float:
    """Aggregate NMSE (Eq. (8)) over an entire dataset."""
    model.eval()
    tot_err = tot_ref = 0.0
    for batch in loader:
        pred = _forward(model, batch, device)
        err, ref = nmse_terms(pred, batch["target"].to(device))
        tot_err += err.item()
        tot_ref += ref.item()
    return tot_err / (tot_ref + 1e-12)
