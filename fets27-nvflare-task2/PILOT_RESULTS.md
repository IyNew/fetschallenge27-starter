# Task 2 Centralized Pilot Results

Reference record of centralized classifier training for the FeTS27 Task 2
brain-tumor task (glioma `0` vs meningioma `1`, four MRI modalities
`t1c`/`t1n`/`t2f`/`t2w`). It holds two things:

1. the Task 2 centralized baseline produced in this repository, and
2. the results previously recorded by the original classifier pilot, kept here
   for comparison.

Every number is recorded inline, so this document stays valid even though the
run artifacts under `outputs/` are gitignored and may be deleted.

**Standing caveat.** All figures below are computed on the official 395-case
validation split, which is also the split used to select the best checkpoint.
They are therefore validation-tuned estimates, not untouched-test results. The
same caveat applied to the pilot's recorded numbers.

## Metric definitions

| Metric | Definition |
|---|---|
| `balanced_accuracy` | mean of sensitivity and specificity at a fixed 0.5 threshold (the key metric used for checkpoint selection) |
| `accuracy` | fraction of correctly classified validation cases |
| `auroc` | rank-based area under the ROC curve |
| `macro_f1` | unweighted mean of the positive- and negative-class F1 scores |
| `sensitivity` | recall of the meningioma class |
| `specificity` | recall of the glioma class |
| `mcc` | Matthews correlation coefficient; reported as `0.0` when a confusion-matrix marginal is zero |
| `confusion_matrix` | `[[tn, fp], [fn, tp]]` |

## Task 2 centralized baseline (current run)

Run name: `central-resnet50-96cubed`. Date: 2026-10-16.

### Setup

| Item | Value |
|---|---|
| Data root | `./data/fets2027_task2` (built from `fets2027-data` by `scripts/prepare_partition_datalists.py`) |
| Split | `site=site-All` → 2,332 training cases (1311 glioma / 1021 meningioma); official 395-case validation split (254 / 141) |
| Model | `resnet50_3d` — BatchNorm 3D ResNet-50, 46,278,199 parameters |
| Input | 96 × 96 × 96, four channels, MONAI crop-foreground → resample → normalize pipeline |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-5, cosine annealing to 1e-6 over 20 epochs |
| Loss | weighted cross-entropy, inverse-frequency class weights |
| Batch | 4 train / 4 validation, AMP enabled, seed 17 |
| Caching | `cache_rate 0.6` (≈23 GB), 12 cache workers, 12 loader workers |
| Hardware | NVIDIA RTX A6000 48 GB under WSL2; peak 5,342 MiB VRAM |
| Wall clock | ≈62 min (≈13 min dataset caching + 20 epochs at 2.5–2.8 min each, 3,135 s of logged epoch time) |

Early stopping (patience 6) never triggered: all 20 epochs ran, and the best
checkpoint came from epoch 15.

### Reproduce

```bash
cd fets27-nvflare-task2
python scripts/prepare_partition_datalists.py \
  --training-partition-csv /path/to/fets2027-data/Task2_Classification_Training_Partition.csv \
  --validation-labels-csv  /path/to/fets2027-data/Task2_Classification_Validation_Labels.csv \
  --training-dir  /path/to/fets2027-data/training \
  --validation-dir /path/to/fets2027-data/validation \
  --data-root     ./data/fets2027_task2

PYTHONPATH=src python -m fets27_classification.cli train \
  --config configs/models/resnet50.yaml \
  --run-name central-resnet50-96cubed \
  --override data.cache_rate=0.6 \
  --override data.cache_workers=12 \
  --override data.num_workers=12
```

### Best model (epoch 15)

| Metric | Value |
|---|---|
| balanced accuracy | **0.9299** |
| accuracy | 0.9241 |
| AUROC | 0.9773 |
| macro-F1 | 0.9192 |
| sensitivity (meningioma) | 0.9504 |
| specificity (glioma) | 0.9094 |
| MCC | 0.8418 |
| confusion matrix | `[[231, 23], [7, 134]]` |

### Per-epoch history

| Epoch | Train loss | Accuracy | Balanced acc. | AUROC | Macro-F1 | Sensitivity | Specificity | MCC | Epoch s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5991 | 0.7190 | 0.7152 | 0.8091 | 0.7050 | 0.7021 | 0.7283 | 0.4172 | 169.8 |
| 2 | 0.4860 | 0.8456 | 0.8547 | 0.9138 | 0.8383 | 0.8865 | 0.8228 | 0.6864 | 161.4 |
| 3 | 0.4664 | 0.7975 | 0.8252 | 0.9229 | 0.7935 | 0.9220 | 0.7283 | 0.6232 | 159.0 |
| 4 | 0.4243 | 0.8076 | 0.8346 | 0.9297 | 0.8035 | 0.9291 | 0.7402 | 0.6413 | 155.0 |
| 5 | 0.3704 | 0.8861 | 0.8830 | 0.9558 | 0.8776 | 0.8723 | 0.8937 | 0.7562 | 154.5 |
| 6 | 0.3418 | 0.8658 | 0.8689 | 0.9475 | 0.8578 | 0.8794 | 0.8583 | 0.7200 | 153.6 |
| 7 | 0.2999 | 0.8684 | 0.8787 | 0.9397 | 0.8620 | 0.9149 | 0.8425 | 0.7334 | 151.1 |
| 8 | 0.2738 | 0.9013 | 0.8980 | 0.9573 | 0.8936 | 0.8865 | 0.9094 | 0.7879 | 155.1 |
| 9 | 0.2588 | 0.8835 | 0.8984 | 0.9628 | 0.8784 | 0.9504 | 0.8465 | 0.7695 | 156.0 |
| 10 | 0.2157 | 0.9215 | 0.9122 | 0.9702 | 0.9141 | 0.8794 | 0.9449 | 0.8283 | 156.9 |
| 11 | 0.1925 | 0.9114 | 0.9153 | 0.9676 | 0.9056 | 0.9291 | 0.9016 | 0.8141 | 160.9 |
| 12 | 0.1711 | 0.8608 | 0.8807 | 0.9620 | 0.8560 | 0.9504 | 0.8110 | 0.7318 | 158.7 |
| 13 | 0.1430 | 0.9114 | 0.9106 | 0.9646 | 0.9048 | 0.9078 | 0.9134 | 0.8107 | 153.2 |
| 14 | 0.1496 | 0.8987 | 0.9118 | 0.9737 | 0.8938 | 0.9574 | 0.8661 | 0.7975 | 154.8 |
| **15** | 0.1234 | **0.9241** | **0.9299** | 0.9773 | **0.9192** | 0.9504 | 0.9094 | **0.8418** | 156.1 |
| 16 | 0.0957 | 0.9165 | 0.9193 | 0.9784 | 0.9107 | 0.9291 | 0.9094 | 0.8237 | 152.7 |
| 17 | 0.0983 | 0.9089 | 0.9149 | 0.9758 | 0.9033 | 0.9362 | 0.8937 | 0.8107 | 155.2 |
| 18 | 0.0864 | 0.9038 | 0.9126 | 0.9790 | 0.8984 | 0.9433 | 0.8819 | 0.8030 | 153.4 |
| 19 | 0.0707 | 0.9114 | 0.9201 | 0.9765 | 0.9063 | 0.9504 | 0.8898 | 0.8183 | 152.1 |
| 20 | 0.0691 | 0.9215 | 0.9279 | 0.9767 | 0.9166 | 0.9504 | 0.9055 | 0.8370 | 165.7 |

Training loss fell steadily (0.5991 → 0.0691) while validation balanced accuracy
plateaued from epoch 15 onward (0.9299 → 0.9279 by epoch 20), which is why the
best checkpoint is epoch 15 rather than the last epoch.

### Artifacts

All under `outputs/central-resnet50-96cubed/` (gitignored):

| File | Contents |
|---|---|
| `summary.json` | best-epoch metrics, data manifest, best epoch index |
| `history.json` | one full metric record per epoch |
| `epoch_metrics.jsonl` | the same records appended live during training |
| `best.pt` | best checkpoint; 555 MB because it also stores optimizer and scheduler state (weights alone are ≈185 MB) |
| `best_predictions.csv` | per-case validation predictions and meningioma probability |
| `config.json` | fully resolved run configuration |
| `train.log` | run log |
| `tensorboard/` | per-epoch scalar summaries |

## Previously recorded classifier-pilot results

Provenance: `fets27-classifier-pilot/PILOT_STATUS.md`, updated 2026-09-15. The
original pilot trained on the integrated labels CSV with 2,311 training cases
and 395 validation cases.

### Centralized baseline (pilot)

| Model | Run | Best validation result | Status |
|---|---|---|---|
| MONAI 3D ResNet-50 | `resnet-local-screen-01` | BA 0.9705, AUROC 0.9943 at epoch 19 | Completed |

### Historical federated results (pilot)

These used the former 33-center, all-pure partition and, as the pilot notes, are
not comparable to the integrated-CSV baseline.

| Screen | Best setting | Best validation BA | AUROC | Status |
|---|---|---:|---:|---|
| GroupNorm ResNet + FedProx | prior enhanced baseline | 0.5815 | 0.5100 | Completed |
| FedLC LR screen | `tau=1`, LR `1e-4`, 4 local batches | 0.5935 | 0.5946 | Completed |
| FedLC tau diagnostic | `tau=0.05`, 4 local batches | 0.5615 | 0.5702 | Completed |
| Seed-matched local-drift screen | `tau=1`, 2 local batches, LR `1e-4`, seed 1901 | 0.6109 | 0.6384 | Completed |

The pilot recorded that all its federated runs used class-stratified
aggregation: sample-weighted averaging within each label group, followed by an
equal 50/50 mean of the two label-group updates. Its best federated figure is a
one-seed pilot measurement and is not a final estimate.

## Comparability notes

- **Training set.** This Task 2 repository trains on 2,332 cases; the pilot's
  recorded centralized run used 2,311. The released
  `Task2_Classification_Training_Partition.csv` adds 21 site-0 cases.
- **Selection.** The pilot's 0.9705 is a screen winner over an
  lr × weight-decay × dropout grid, whereas the current figure is a single run at
  the config's default learning rate (3e-4). The two are therefore not a
  like-for-like reproduction.
- **Same selection split.** Both select on the same 395-case validation split, so
  both are validation-tuned estimates rather than untouched-test results.
- The current 0.9299 balanced accuracy / 0.9773 AUROC is in the same range as the
  pilot's 0.9705 / 0.9943 but lower, consistent with the differences above.

## Other Task 2 runs (pipeline checks only)

Neither of these is a performance result; both use the synthetic toy root from
`prepare-assets` and exist only to prove the training and federation paths run
end to end.

- Centralized toy run (`toy-smoke`): 1 epoch, 4 synthetic training cases, 2
  synthetic validation cases, balanced accuracy 0.5.
- Federated NVFLARE smoke: 2 sites, 1 round, 2 synthetic validation cases,
  balanced accuracy 0.5.
