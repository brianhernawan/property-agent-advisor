# Checkpoint 2 — Experiment Results

Property Due-Diligence & Investment Advisor — building damage CNN (xBD)

## Experiment log

Sorted by stage (A → C), best validation macro-F1 first within each stage.

| Stage | Run | Model | Unfreeze | LR | Batch | Balance | Val macro-F1 |
|---|---|---|---|---|---|---|---|
| A | A3_r18_full | resnet18 | full | 1e-4 | 64 | weights | 0.6456 |
| A | A2_r18_last | resnet18 | last | 1e-4 | 64 | weights | 0.6307 |
| A | A1_r18_head | resnet18 | head | 1e-3 | 64 | weights | 0.5290 |
| B | B3_sqrtw | resnet18 | last | 1e-4 | 64 | sqrt_weights | 0.6738 |
| B | B4_sampler | resnet18 | last | 1e-4 | 64 | sampler | 0.6312 |
| B | B1_bs32 | resnet18 | last | 1e-4 | 32 | weights | 0.6292 |
| B | B2_bs128 | resnet18 | last | 1e-4 | 128 | weights | 0.6249 |
| B | B5_focal | resnet18 | last | 1e-4 | 64 | weights (focal loss) | 0.6217 |
| C | C2_resnet50 ★ | resnet50 | last | 1e-4 | 64 | sqrt_weights | 0.6934 |
| C | C1_resnet18 | resnet18 | last | 1e-4 | 64 | sqrt_weights | 0.6925 |
| C | C3_effb0 | efficientnet_b0 | last | 1e-4 | 64 | sqrt_weights | 0.6695 |

★ registered as champion

## Registered model

- Model: `proptech-damage-cnn` v1, alias `champion`
- Source run: C2_resnet50 (run id `5f8212a0eb574cbf834734feeaf4b22a`)
- Validation macro-F1: 0.6934

## Test-set evaluation (champion, n=24,080)

```
test split, n=24,080
accuracy 0.8515   macro-F1 0.7274
```

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no-damage | 0.954 | 0.908 | 0.931 | 17,725 |
| minor-damage | 0.469 | 0.593 | 0.524 | 2,247 |
| major-damage | 0.626 | 0.648 | 0.637 | 2,124 |
| destroyed | 0.787 | 0.853 | 0.818 | 1,984 |
| **accuracy** | | | **0.852** | 24,080 |
| macro avg | 0.709 | 0.751 | 0.727 | 24,080 |
| weighted avg | 0.866 | 0.852 | 0.857 | 24,080 |

## Confusion

- minor-damage → major-damage: 21.8%
- major-damage → minor-damage: 17.7%

Logged to MLflow run `5f8212a0eb574cbf834734feeaf4b22a`; full artifacts saved to `eval_out/C2_resnet50/`.

## What the columns mean

| Term | Meaning |
|---|---|
| Run | The name given to that training run, used to find it in MLflow and in `checkpoints/<run>/` |
| Model | Which CNN architecture: resnet18, resnet50, efficientnet_b0 — bigger = more capacity, slower |
| Unfreeze | How much of the pretrained network was allowed to retrain: head (new classifier only, fastest), last (classifier + last block), full (entire network, slowest, most adaptable) |
| LR | Learning rate — how big a step the optimizer takes each update. Too high = unstable, too low = learns too slowly |
| Batch | How many image patches are averaged per gradient update. Bigger = smoother/more stable, more memory |
| Balance | How the 73%-no-damage class imbalance was handled: weights (penalize rare-class mistakes more), sqrt_weights (same, gentler), sampler (oversample rare classes), focal (loss that focuses on hard examples) |
| Val macro-F1 | F1 averaged equally across all 4 classes on validation data — the number used to pick the winner. Higher is better; unweighted so a model can't win just by nailing the easy majority class |

## What precision, recall, F1, support mean

- **Precision** — of everything the model called this class, what fraction actually was. Low precision = lots of false alarms.
- **Recall** — of everything that actually is this class, what fraction the model caught. Low recall = lots of misses.
- **F1** — the balance of precision and recall in one number. Only high when both are reasonably good.
- **Support** — how many real test examples of that class existed (a count, not a score).

## Reading the results: good, bad, and why

- **no-damage (0.954 / 0.908 / 0.931) — good.** Largest class (17,725 examples), visually distinct, easiest to learn.
- **destroyed (0.787 / 0.853 / 0.818) — good.** Smallest class but visually unambiguous (rubble is rubble), so the model still does well despite less data.
- **major-damage (0.626 / 0.648 / 0.637) — mediocre.**
- **minor-damage (0.469 / 0.593 / 0.524) — the weak point**, specifically its precision: the model over-calls "minor-damage" on things that are actually major-damage (or vice versa). This lines up exactly with the confusion numbers above — minor and major are the two rarest classes AND the two that sit right next to each other on the damage scale, so they're the easiest for the model to mix up.

How to push minor/major higher:
1. **Try full unfreeze on the final architecture run, not just last-block.** Stage A showed full (0.646) beat last (0.631) by a real margin on the same resnet18 — stage C used "last" to save compute. Re-running C2/C1 with full unfreeze is the single most direct lever left untried.
2. **Combine sampler + sqrt_weights instead of either alone.** B3 (sqrt_weights) beat B4 (sampler) on its own, but stacking both could push more minor/major examples through training without sqrt_weights' softer touch being the only correction.
3. **Treat damage as ordinal, not just categorical.** Right now a minor↔major mistake is penalized the same as a no-damage↔destroyed mistake, even though the former is a much smaller error. An ordinal loss (or at least a confusion-aware weighting) would stop the model being "equally wrong" about adjacent vs. opposite classes.
4. **More epochs / harder augmentation targeted at the minor/major boundary**, since both are minority classes and the model may simply be underfit on them relative to no-damage.
