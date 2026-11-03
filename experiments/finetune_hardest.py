"""
Stage 2a - one-shot hard-example fine-tuning.

Starting from the base model of a run directory (see ``train_base.py``), keep
only the ``N`` training examples with the highest base-model loss and continue
training on that subset for ``--epochs`` more epochs. The control ("baseline")
continues training the *same* base model for the same number of epochs on the
full training set. Every subset size starts from an identical copy of the base
model, so the curves are directly comparable.

For each ``N`` a plot ``plots/hardest_<N>.png`` overlays the validation
accuracy of the pruned run (solid) and of the baseline (dashed) - the format of
the figures in ``results/onn_mnist`` and ``results/ann_*``. A summary table
``finetune_summary.csv`` and ``plots/summary_final_val_accuracy.png`` collect
the final accuracies.

Example::

    python -m experiments.finetune_hardest --run runs/onn_mnist --epochs 20 \\
        --subset-sizes 1000 2000 3000 4000 5000 10000 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import common as C

DEFAULT_SIZES = (1000, 2000, 3000, 4000, 5000, 10000, 20000)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run directory written by train_base.py")
    p.add_argument("--epochs", type=int, default=20, help="fine-tuning epochs (pruned runs and baseline)")
    p.add_argument("--batch-size", type=int, default=None, help="defaults to the base run's batch size")
    p.add_argument("--subset-sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                   help="numbers N of hardest examples to keep")
    p.add_argument("--ranking", type=Path, default=None,
                   help="alternative idx,loss (or legacy single-column) CSV to rank examples with")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def continue_training(problem: C.Problem, run: Path, indices: np.ndarray, epochs: int,
                      batch_size: int) -> dict:
    """Load a fresh copy of the base model and train it on ``x_train[indices]``."""
    model = C.load_model(run / "base_model")
    history = model.fit(problem.x_train[indices], problem.y_train[indices],
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(problem.x_test, problem.y_test), verbose=2)
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def main(argv=None) -> None:
    args = parse_args(argv)
    C.set_seed(args.seed)

    run = args.run
    cfg = C.load_config(run / "config.json")
    batch_size = args.batch_size or cfg["batch_size"]
    problem = C.problem_from_config(cfg)
    ranking = C.load_ranking(args.ranking or run / "loss_per_example.csv")
    tag = f"{args.epochs}eph"
    plots = run / "plots"

    # ---- baseline: keep training on everything --------------------------------
    baseline_path = run / f"baseline_{tag}_history.pkl"
    if baseline_path.exists():
        baseline = C.load_history(baseline_path)
        print(f"[finetune] reusing baseline {baseline_path}")
    else:
        print(f"[finetune] baseline: {args.epochs} more epochs on all {problem.n_train} examples")
        baseline = continue_training(problem, run, np.arange(problem.n_train), args.epochs, batch_size)
        C.save_history(baseline, baseline_path)

    # ---- pruned runs -----------------------------------------------------------
    rows = []
    for n in args.subset_sizes:
        n = min(n, problem.n_train)
        keep = C.hardest(ranking, n)
        print(f"[finetune] {args.epochs} epochs on the {n} hardest examples")
        hist = continue_training(problem, run, keep, args.epochs, batch_size)
        C.save_history(hist, run / f"hardest_{n}_{tag}_history.pkl")

        C.plot_val_accuracy(
            {"val_accuracy_pruning": hist["val_accuracy"],
             "val_accuracy_baseline": baseline["val_accuracy"]},
            title=f"{problem.dataset}: {cfg['epochs']}eph base, then {args.epochs}eph on the {n} hardest",
            out_png=plots / f"hardest_{n}.png", dashed=("val_accuracy_baseline",), ylim=(0, 1))
        rows.append(dict(subset_size=n, fraction=n / problem.n_train,
                         final_val_accuracy=hist["val_accuracy"][-1],
                         best_val_accuracy=max(hist["val_accuracy"]),
                         baseline_final_val_accuracy=baseline["val_accuracy"][-1]))

    # ---- summary ---------------------------------------------------------------
    summary = pd.DataFrame(rows).sort_values("subset_size")
    summary.to_csv(run / f"finetune_summary_{tag}.csv", index=False)
    print(summary.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(summary["subset_size"], summary["final_val_accuracy"], marker="o", label="hardest-N subset")
    ax.axhline(baseline["val_accuracy"][-1], linestyle="--", color="tab:orange", label="full training set")
    ax.set_xscale("log")
    ax.set_xlabel("N hardest examples kept")
    ax.set_ylabel(f"val_accuracy after {args.epochs} epochs")
    ax.set_title(f"{problem.kind.upper()} on {problem.dataset}: hard-example fine-tuning")
    ax.legend()
    fig.tight_layout()
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / f"summary_final_val_accuracy_{tag}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
