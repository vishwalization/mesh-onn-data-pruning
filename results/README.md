# Results

Validation-accuracy curves from the 2023 runs. Every plot overlays two curves:

* **`val_accuracy_pruning`** (solid) – the model trained on a loss-selected subset,
* **`val_accuracy_baseline`** (dashed) – the reference trained on the full set.

`overview.png` is a 2×2 montage of the four plots used in the top-level README.

| Folder | Model | Dataset | What was varied | Reference curve |
|---|---|---|---|---|
| `onn_mnist/base_5eph/` | 2-layer, 256-mode Clements-mesh ONN | MNIST | base model trained **5** epochs on all 60k, then **20** epochs on the hardest *N* ∈ {1k, 2k, 3k, 4k, 5k, 10k, 20k} | the same base model trained 20 more epochs on all 60k |
| `onn_mnist/base_2eph/` | same ONN | MNIST | as above with a **2**-epoch base model | same |
| `ann_fashion_mnist/` | MLP 784-165-10 | Fashion-MNIST | 5-epoch base, then 20 epochs on the hardest *N* ∈ {1k … 60k} | base model trained 20 more epochs on all 60k |
| `ann_mnist/` | MLP 784-165-10 | MNIST | as above | same |
| `gradual_pruning_mnist/retention_{30,50,80}/` | MLP 784-165-10 | MNIST | iterative pruning: each round keeps the hardest 30 / 50 / 80 % of the previous round's set and trains 5 epochs | first 5 epochs of a from-scratch run on all 60k |
| `archive/ann_fashion_mnist_val_loss_oct2023/` | MLP | Fashion-MNIST | an earlier sweep that plotted validation **loss** instead of accuracy | — |

File names encode the setting, e.g. `mnist_history_5_eph_20_eph_on_top_5000hard.png`
= 5-epoch base model, 20 fine-tuning epochs on the 5 000 hardest examples;
`retention_80/keep_38400_of_48000.png` = the round that kept 38 400 of the previous 48 000.

## Reading the plots

* **ONN, one-shot pruning** – with the hardest 20k (a third of MNIST) the pruned
  mesh tracks the full-data baseline epoch for epoch; with 10k it is still within
  a percent; below ~5k the gap opens and the curve drifts downward with more
  epochs (the hardest few thousand examples are a biased, noisy slice of the data).
* **MLP, gradual pruning** – shrinking the set geometrically while re-ranking by
  the *current* model's loss keeps accuracy flat at ≈ 0.97 all the way down to a
  few thousand images at 80 % retention. The 30 % and 50 % schedules collapse the
  set much faster and are included to show where the approach breaks down
  (sets of tens of images). Note that in these plots the dashed reference is a
  model trained *from scratch* for 5 epochs, whereas the pruned model starts from
  the 5-epoch base model – read them as "pruning does not hurt", not as a
  like-for-like speed comparison.
* **MLP, one-shot pruning** – for the dense network the one-shot variant is far
  more fragile than for the ONN; see the caveat below before quoting the
  `ann_mnist/` numbers.

## Caveat on `ann_mnist/`

The archived notebook that produced the `ann_mnist/` sweep ends in a state where
the fine-tuning subsets are drawn from Fashion-MNIST while the loss ranking and
the dashed baseline come from MNIST. The plots were therefore not regenerated for
this repository as-is and should be reproduced with
`experiments/finetune_hardest.py` (which cannot mix datasets) before being
quoted. The `onn_mnist/`, `ann_fashion_mnist/` and `gradual_pruning_mnist/`
figures are unaffected.

## Regenerating

```bash
python -m experiments.train_base       --model onn --dataset mnist --epochs 5 --out runs/onn_mnist
python -m experiments.finetune_hardest --run runs/onn_mnist --epochs 20 --subset-sizes 1000 2000 3000 4000 5000 10000 20000

python -m experiments.train_base       --model ann --dataset mnist --epochs 5 --out runs/ann_mnist
python -m experiments.gradual_pruning  --run runs/ann_mnist --retention 0.8 --rounds 10 --epochs 5
```

Trained checkpoints (TensorFlow SavedModels, ≈ 220 MB) are not tracked in git;
the scripts above recreate them in `runs/`.
