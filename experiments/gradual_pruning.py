"""
Stage 2b - gradual (iterative) pruning.

Instead of throwing away most of the data in one go, shrink the training set
geometrically. Starting from the base model and its loss ranking:

    round 0:  S_0 = all N examples
    round i:  rank S_{i-1} by the *current* model's loss, keep the hardest
              r * |S_{i-1}| examples  ->  S_i,  train ``--epochs`` more epochs on S_i,
              re-score S_i with the updated model.

``r`` is ``--retention`` (0.3, 0.5 and 0.8 in ``results/gradual_pruning_mnist``).
Subset sizes follow ``int(r**i * N)``, e.g. for N = 60000, r = 0.8:
60000 -> 48000 -> 38400 -> 30720 -> ...; the model is carried over from round to
round, so each round starts from the previous round's weights.

Each round is plotted against a *from-scratch* reference curve trained for
the same number of epochs on the full set (``plots/round_<i>_keep_<n>_of_<m>.png``).
By default that reference is the base run itself (it is a from-scratch run on
all data); pass ``--baseline-history`` to overlay another history pickle
instead, as the original notebooks did with a 50-epoch baseline.

Example::

    python -m experiments.gradual_pruning --run runs/ann_mnist --retention 0.8 --rounds 10 --epochs 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import common as C


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run directory written by train_base.py")
    p.add_argument("--retention", type=float, required=True, help="fraction r of the current set kept per round")
    p.add_argument("--rounds", type=int, default=10, help="maximum number of pruning rounds")
    p.add_argument("--epochs", type=int, default=5, help="training epochs per round")
    p.add_argument("--batch-size", type=int, default=None, help="defaults to the base run's batch size")
    p.add_argument("--baseline-history", type=Path, default=None,
                   help="history pickle to use as the dashed reference curve (default: the base run)")
    p.add_argument("--save-models", action="store_true", help="also save the model after every round")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def schedule(n_total: int, retention: float, rounds: int) -> list:
    """Subset sizes ``int(r**i * N)`` for i = 0..rounds, truncated when a set would be empty."""
    sizes = []
    for i in range(rounds + 1):
        n = int(np.power(retention, i) * n_total)
        if n < 1:
            break
        sizes.append(n)
    return sizes


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.retention < 1.0:
        raise SystemExit("--retention must be strictly between 0 and 1")
    C.set_seed(args.seed)

    run = args.run
    cfg = C.load_config(run / "config.json")
    batch_size = args.batch_size or cfg["batch_size"]
    problem = C.problem_from_config(cfg)

    pct = int(round(args.retention * 100))
    out = run / f"gradual_pruning_retention{pct}"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # reference curve: a from-scratch run on the full set, first `epochs` epochs
    baseline_path = args.baseline_history or run / "base_history.pkl"
    baseline = C.load_history(baseline_path)["val_accuracy"][:args.epochs]
    if len(baseline) < args.epochs:
        print(f"[gradual] note: reference history {baseline_path} has only {len(baseline)} epochs")

    sizes = schedule(problem.n_train, args.retention, args.rounds)
    print(f"[gradual] retention {args.retention}: subset sizes {sizes}")

    model = C.load_model(run / "base_model")
    ranking = C.load_ranking(run / "loss_per_example.csv")  # ranking of S_0 under the base model
    rows = []

    for i in range(1, len(sizes)):
        n_prev, n_keep = sizes[i - 1], sizes[i]
        keep = C.hardest(ranking, n_keep)
        print(f"[gradual] round {i}: keeping the hardest {n_keep} of {n_prev} ({pct}%), "
              f"{args.epochs} epochs")

        history = model.fit(problem.x_train[keep], problem.y_train[keep],
                            epochs=args.epochs, batch_size=batch_size,
                            validation_data=(problem.x_test, problem.y_test), verbose=2)
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        C.save_history(hist, out / f"round_{i:02d}_keep_{n_keep}_history.pkl")
        if args.save_models:
            C.save_model(model, out / f"round_{i:02d}_keep_{n_keep}_model")

        # re-score the kept examples with the updated model -> ranking for the next round
        losses = C.per_example_loss(model, problem.x_train[keep], problem.y_train[keep])
        ranking = C.rank_by_loss(losses, indices=keep)
        C.save_ranking(ranking, out / f"round_{i:02d}_keep_{n_keep}_loss_per_example.csv")

        C.plot_val_accuracy(
            {"val_accuracy_pruning": hist["val_accuracy"],
             "val_accuracy_baseline": baseline},
            title=f"Gradual pruning: keeping the hardest {n_keep} of {n_prev} ({pct}%)",
            out_png=plots / f"round_{i:02d}_keep_{n_keep}_of_{n_prev}.png",
            dashed=("val_accuracy_baseline",))
        rows.append(dict(round=i, kept=n_keep, previous=n_prev, fraction_of_full=n_keep / problem.n_train,
                         final_val_accuracy=hist["val_accuracy"][-1],
                         reference_final_val_accuracy=baseline[-1] if baseline else float("nan")))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
