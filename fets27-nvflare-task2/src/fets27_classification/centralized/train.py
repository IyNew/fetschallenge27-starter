"""Centralized training, validation, and checkpoint orchestration."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from ..metrics import binary_metrics
from ..models import build_model
from .data import audit_cases, build_loaders, load_cases, seed_everything

LOGGER = logging.getLogger(__name__)

CHECKPOINT_NAME = "best.pt"
RESUME_REQUIRED_KEYS = ("model", "optimizer", "scheduler", "epoch", "best_score")


def run_training(
    config: dict[str, Any], run_name: str, resume: str | None = None
) -> dict[str, Any]:
    """Train one configured model and return its best validation summary.

    The best checkpoint is selected by validation balanced accuracy at the fixed
    0.5 decision threshold. Accuracy, AUROC, macro-F1, sensitivity, specificity,
    MCC, and the confusion matrix are recorded for every epoch.

    Args:
        config: Loaded configuration mapping.
        run_name: Name of the output subdirectory under ``config['output_dir']``.
        resume: Optional checkpoint path to continue training from.

    Returns:
        The summary dictionary written to ``summary.json``.
    """
    torch = _torch()
    from torch.utils.tensorboard import SummaryWriter

    seed_everything(int(config["training"]["seed"]))
    device = _device(config["training"].get("device", "auto"))
    output_dir = Path(config["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(config)
    train_loader, valid_loader, class_weights = build_loaders(cases, config)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(config["training"].get("min_learning_rate", 1e-6)),
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    amp_enabled = device.type == "cuda" and bool(config["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_score, stale = 1, float("-inf"), 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        missing = [key for key in RESUME_REQUIRED_KEYS if key not in checkpoint]
        if missing:
            raise ValueError(f"Resume checkpoint is missing keys: {missing}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
    writer = SummaryWriter(output_dir / "tensorboard")
    history: list[dict[str, Any]] = []
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        epoch_start = perf_counter()
        _reset_peak_memory(torch, device)
        train_loss, train_seconds, train_samples = _train_epoch(
            torch, model, train_loader, optimizer, criterion, scaler, device, amp_enabled
        )
        validation_start = perf_counter()
        metrics, rows, validation_samples = _validate(
            torch, model, valid_loader, device, amp_enabled
        )
        validation_seconds = perf_counter() - validation_start
        scheduler.step()
        epoch_seconds = perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds,
            "train_samples_per_second": train_samples / max(train_seconds, 1e-9),
            "validation_samples_per_second": validation_samples
            / max(validation_seconds, 1e-9),
            "peak_gpu_memory_mib": _peak_memory_mib(torch, device),
            **metrics,
        }
        history.append(record)
        for key, value in record.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch)
        writer.flush()
        _append_epoch_record(epoch_log_path, record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info(
            "epoch=%s loss=%.6f balanced_accuracy=%.6f auroc=%.6f mcc=%.6f "
            "train_seconds=%.1f validation_seconds=%.1f epoch_seconds=%.1f "
            "train_samples_per_second=%.2f validation_samples_per_second=%.2f "
            "peak_gpu_memory_mib=%.0f",
            epoch,
            train_loss,
            metrics["balanced_accuracy"],
            metrics["auroc"],
            metrics["mcc"],
            train_seconds,
            validation_seconds,
            epoch_seconds,
            record["train_samples_per_second"],
            record["validation_samples_per_second"],
            record["peak_gpu_memory_mib"],
        )
        if metrics["balanced_accuracy"] > best_score:
            best_score, stale = float(metrics["balanced_accuracy"]), 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_score": best_score,
                    "config": config,
                },
                output_dir / CHECKPOINT_NAME,
            )
            _write_predictions(output_dir / "best_predictions.csv", rows)
        else:
            stale += 1
        if stale >= int(config["training"].get("early_stopping_patience", 999)):
            break
    writer.close()
    best_checkpoint = torch.load(
        output_dir / CHECKPOINT_NAME, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    best_metrics, _, _ = _validate(torch, model, valid_loader, device, amp_enabled)
    summary = {
        "run_name": run_name,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_checkpoint["epoch"],
        "manifest": audit_cases(cases),
        "metrics": best_metrics,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _train_epoch(
    torch, model, loader, optimizer, criterion, scaler, device, amp_enabled: bool
) -> tuple[float, float, int]:
    model.train()
    total = 0.0
    samples = 0
    started = perf_counter()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(_logits(model(image)), label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        samples += int(label.shape[0])
    _synchronize(torch, device)
    return total / max(len(loader), 1), perf_counter() - started, samples


def _validate(torch, model, loader, device, amp_enabled: bool):
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    rows = []
    samples = 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = _logits(model(batch["image"].to(device, non_blocking=True)))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            truth = batch["label"].long().tolist()
            labels.extend(int(value) for value in truth)
            probabilities.extend(float(value) for value in prob)
            samples += len(truth)
            rows.extend(
                {
                    "subject": subject,
                    "label": int(label),
                    "meningioma_probability": float(value),
                }
                for subject, label, value in zip(batch["subject"], truth, prob)
            )
    _synchronize(torch, device)
    return binary_metrics(labels, probabilities), rows, samples


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["subject", "label", "meningioma_probability"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _device(value: str):
    torch = _torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _logits(output):
    """Normalize MONAI classifier outputs to the primary logits tensor."""
    return output[0] if isinstance(output, tuple) else output


def _append_epoch_record(path: Path, record: dict[str, Any]) -> None:
    """Append one immediately visible, JSON-encoded epoch performance record."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _reset_peak_memory(torch, device) -> None:
    """Reset CUDA peak allocation tracking at the start of an epoch."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mib(torch, device) -> float:
    """Return the epoch's maximum CUDA allocation in MiB, or zero on CPU."""
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def _synchronize(torch, device) -> None:
    """Synchronize CUDA before recording wall-clock performance measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _torch():
    import torch

    return torch
