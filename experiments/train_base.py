"""
Stage 1 - train a *base* model on the full training set and score every
training example by its loss.

This is the common starting point of both pruning experiments:

1. train a fresh model for a few epochs on all N training images
   (``--epochs 5`` reproduces the "5eph" base models in ``results/``),
2. compute the categorical cross-entropy of each training example under that
   model and save the ranking ``loss_per_example.csv`` (``idx, loss`` sorted by
   decreasing loss - the "hardest" examples come first).

Outputs (inside ``--out``)::

    config.json            what was trained (model kind, dataset, sizes, epochs)
    base_model/            SavedModel (or base_model.keras under Keras 3)
    base_history.pkl       Keras history dict of the base run
    loss_per_example.csv   idx,loss ranking of the whole training set

Example::

    python -m experiments.train_base --model onn --dataset mnist --epochs 5 --out runs/onn_mnist
    python -m experiments.train_base --model ann --dataset fashion_mnist --epochs 5 --out runs/ann_fashion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments import common as C


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=C.MODEL_KINDS, required=True, help="dense ANN control or mesh ONN")
    p.add_argument("--dataset", choices=C.DATASETS, default="mnist")
    p.add_argument("--epochs", type=int, default=5, help="base-training epochs on the full set")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", type=Path, required=True, help="run directory to create")
    p.add_argument("--seed", type=int, default=None)
    # ONN geometry
    p.add_argument("--windowhalfwidth", type=int, default=8,
                   help="half-width of the Fourier low-pass window; inputs = (2*w)^2 = 256 by default")
    p.add_argument("--layers", type=int, default=2, help="number of mesh layers (ONN)")
    p.add_argument("--bs-error", type=float, default=0.0, help="beamsplitter error of the simulated mesh (ONN)")
    # ANN geometry
    p.add_argument("--hidden", type=int, default=165, help="hidden units of the dense ANN")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    C.set_seed(args.seed)

    cfg = dict(model=args.model, dataset=args.dataset, epochs=args.epochs,
               batch_size=args.batch_size, seed=args.seed,
               windowhalfwidth=args.windowhalfwidth, layers=args.layers,
               bs_error=args.bs_error, hidden=args.hidden)
    problem = C.problem_from_config(cfg)
    cfg["n_train"] = problem.n_train

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    C.save_config(cfg, out / "config.json")

    print(f"[train_base] {args.model.upper()} on {args.dataset}: "
          f"{problem.n_train} train / {len(problem.x_test)} test, {args.epochs} epochs")
    model = problem.build()
    model.summary()

    history = model.fit(problem.x_train, problem.y_train,
                        epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(problem.x_test, problem.y_test), verbose=2)
    C.save_model(model, out / "base_model")
    C.save_history(history, out / "base_history.pkl")

    print("[train_base] scoring every training example ...")
    losses = C.per_example_loss(model, problem.x_train, problem.y_train)
    ranking = C.rank_by_loss(losses)
    C.save_ranking(ranking, out / "loss_per_example.csv")

    final_acc = history.history["val_accuracy"][-1]
    print(f"[train_base] done. val_accuracy={final_acc:.4f}  "
          f"loss: mean={losses.mean():.4f} median={np.median(losses):.4f} max={losses.max():.3f}")
    print(f"[train_base] hardest 5 training indices: {ranking['idx'].to_numpy()[:5].tolist()}")


if __name__ == "__main__":
    main()
