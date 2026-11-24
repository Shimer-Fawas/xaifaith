# xaifaith — per-molecule faithfulness scoring for molecular XAI

Predicting, for a given molecule and a given explainer, whether the attribution
you are about to show a chemist is faithful to the model.

## The gap this fills

Molecular XAI validation almost always asks whether an explanation matches
chemistry — does it highlight the known toxicophore, does it recover the
structural alert. That measures **plausibility**. Faithfulness to the model is a
different quantity, it is measured far less often, and the one systematic study
of it (Hiltscher, Bianciotto & Grisoni, ChemRxiv 2026) found that faithfulness
varies markedly *per molecule*, not just per method.

Every faithfulness number in the literature is a method-level average. A chemist
deciding about one compound gets no signal about that compound. This repo closes
that: it computes a per-molecule faithfulness label, then trains a model to
predict it from features available before you compute it.

## The metric

For one molecule, mask atoms in the order the attribution ranks them and watch
the model's output collapse. Faster collapse = more faithful attribution. The
problem with using that raw is that a 12-atom fragment and a 40-atom molecule
have different achievable ranges, so raw deletion AUCs are not comparable across
molecules. Each molecule therefore gets its own two reference points:

| reference | meaning | cost |
| --- | --- | --- |
| random order | the floor: what you get knowing nothing | 8 orders x ~20 points |
| greedy oracle | the ceiling: the best ordering that exists for this model and this molecule | O(n²) forwards, exact for n ≤ 60 |

```latex
F_{\text{mol}} = \frac{\mathrm{AUC}_{\text{random}} - \mathrm{AUC}_{\text{attribution}}}{\mathrm{AUC}_{\text{random}} - \mathrm{AUC}_{\text{greedy}}}
```

`F = 1` means the attribution is as good as any ordering could be for this
molecule. `F = 0` means no better than random. `F < 0` means worse than random,
which happens more often than the literature admits — reporting that frequency
per explainer is a result in itself.

Masking replaces atom features with the all-zeros vector, which in this
featurisation means "no information about this atom". That is why
`featurize.py` builds float one-hot features instead of using PyG's integer
encodings: it gives masking a well-defined baseline and gives gradient methods a
continuous input space.

**Degenerate molecules** — where the output barely moves between the full
molecule and the empty graph — are detected and dropped, because no ordering is
distinguishable there. The fraction dropped is itself worth reporting.

## The predictor

Features are constrained to things computable **without** the deletion curves,
or the predictor is useless in deployment:

- **molecule** — RDKit descriptors: size, rings, aromaticity, TPSA, logP, Bertz complexity, branching
- **model** — logit, margin, MC-dropout spread, input-gradient norm, k-NN distance to the training set in embedding space (applicability domain)
- **attribution** — entropy, Gini, top-1 and top-k mass share, sign balance. A smeared attribution and a concentrated one behave differently, and the Gini term is what catches explainers that score well by saying nothing.

LightGBM regressor, evaluated under three protocols of increasing severity:

1. **scaffold-grouped CV** — the baseline number
2. **leave-one-architecture-out** — does it transfer to a backbone it never saw?
3. **leave-one-dataset-out** — does it transfer to new chemistry?

## The claim that matters

Not Spearman. **Gating.** Keep only the explanations the predictor says are
faithful, and report what you actually got:

```
gating at 50% coverage: +0.31 -> +0.58  (oracle +0.74, captured 63% of the available gain)
```

That is the risk-coverage curve in `meta.py`, and it is the number a
practitioner would act on. If it is flat, the paper is a negative result and
still worth publishing.

## Install (Ubuntu / WSL2, RTX 4090)

```bash
bash setup_env.sh
source .venv/bin/activate
python -m xaifaith.smoke_test          # ~1 min, validates the whole pipeline
```

The smoke test builds a synthetic task where nitro-containing molecules are
positive, trains a GIN, and checks that (a) the model learns it, (b) random
attribution scores near F = 0, and (c) integrated gradients beats it on both
faithfulness and nitro recovery. If that table looks wrong, stop before spending
GPU hours.

WSL notes: the GPU driver comes from the Windows host — install only the CUDA
runtime that ships with the torch wheels, never a Linux driver inside WSL. If
`torch.cuda.is_available()` is False, check `nvidia-smi` works inside WSL first.
Keep the repo on the Linux filesystem (`~/...`), not `/mnt/c/...`; dataset I/O
across the 9p mount is roughly an order of magnitude slower.

## Run

```bash
# 1. train backbones (minutes on a 4090)
python -m xaifaith.run train \
  --datasets bace bbbp clintox tox21_nr_ar \
  --architectures gin gat gcn attentivefp

# 2. attributions + faithfulness labels (the expensive stage)
python -m xaifaith.run attribute \
  --datasets bace bbbp clintox tox21_nr_ar \
  --architectures gin gat gcn attentivefp \
  --max-molecules 600

# 3. meta-model + all three protocols
python -m xaifaith.run meta
```

Stage 2 is checkpointed per dataset/architecture and shardable
(`--shard i --num-shards n`), so it resumes after interruption and parallelises
across workers. See `scripts/run_gcp.sh` for the A100 version — you only need it
if you scale to full datasets; the 4090 handles the configuration above.

## Cost model

Per molecule with *n* atoms, per explainer: ~20 masked forwards for the
deletion curve. Per molecule, once (shared across explainers): ~160 for the
random floor, ~n²/2 for the greedy oracle. At n ≈ 25 that is ~500 graph
forwards per molecule, and these graphs are tiny — the 4090 does tens of
thousands of them per second. The greedy oracle dominates, which is why
`--greedy-max-atoms` exists.

## Layout

```
src/xaifaith/
  featurize.py     float one-hot atom/bond features with a zero baseline
  data.py          MoleculeNet loading, deterministic scaffold splits
  models.py        GCN / GIN / GAT / AttentiveFP, shared forward signature
  train.py         training loop, early stopping on scaffold-split validation
  batching.py      K masked copies of one graph in one forward pass
  explain.py       8 attribution methods
  faithfulness.py  the metric: references, greedy oracle, per-molecule F
  features.py      label-free features for the predictor
  meta.py          LightGBM, three protocols, risk-coverage gating
  run.py           CLI
  smoke_test.py    synthetic nitro task, end-to-end check
```

## Design decisions worth defending in review

**Faithfulness is measured on held-out chemistry only.** Measuring it on the
training set conflates "the explanation is faithful" with "the model memorised
this molecule". Scaffold splits throughout, for the same reason: a random split
lets near-duplicate analogues straddle train and test, and any faithfulness
number computed on one is measuring memorisation.

**Occlusion is excluded from the meta-model by default.** It is near-circular
with a deletion-based metric — both mask atom features and watch the output
move. It is computed and reported as a reference point, flagged via
`is_circular_explainer`, and dropped unless you pass `--include-circular`. A
reviewer will look for this; better that it is already handled.

**Sign is information.** Atoms are ranked in the direction of the model's own
prediction, not by absolute magnitude, so an explainer that gets the sign wrong
is penalised rather than accidentally rewarded.

## Extensions, in the order they add the most

1. **B-XAIC and MolExplain** — add ground-truth loaders and report plausibility
   alongside faithfulness for the same molecules. The disagreement between the
   two is the most quotable figure the project can produce.
2. **Conformal calibration** — turn the predictor's output into a coverage
   guarantee rather than a ranking, which is what makes it usable in a workflow.
3. **Regression endpoints** — `esol` and `lipo` are wired up; the metric is
   task-agnostic, and showing it holds outside binary classification widens the
   claim considerably.

## Venue

Built for *Journal of Cheminformatics*, "Evaluating AI and machine learning
models in cheminformatics: benchmarking techniques and case studies". That
collection asks for benchmarking methodology and case studies, which is exactly
what this is. Release the code and the computed label set — the label set is the
reusable artifact and is worth as much as the method.
