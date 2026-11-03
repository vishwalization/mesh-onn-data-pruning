"""
Shared building blocks for the data-pruning experiments.

Everything the three experiment scripts have in common lives here:

* loading a dataset either as raw images (dense-ANN control) or as the
  low-pass-filtered Fourier features that the mesh ONN consumes,
* building the two model families (dense ANN with one hidden layer, and the
  2-layer Clements-mesh ONN from :mod:`mesh_onn.helperfunctions`),
* per-example loss scoring and the "hardest-N" ranking,
* model / history I/O that works with both the SavedModel format used in the
  original runs and the newer ``.keras`` format,
* the validation-accuracy comparison plot used throughout ``results/``.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

DATASETS = ("mnist", "fashion_mnist", "kmnist", "cifar10")
MODEL_KINDS = ("ann", "onn")
N_CLASSES = 10


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: Optional[int]) -> None:
    """Seed numpy and TensorFlow (no-op when ``seed`` is ``None``)."""
    if seed is None:
        return
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:  # TensorFlow < 2.7
        tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _raw_dataset(name: str):
    """``(x_train, y_train), (x_test, y_test)`` with integer labels of shape (N,)."""
    kd = tf.keras.datasets
    if name == "mnist":
        (xtr, ytr), (xte, yte) = kd.mnist.load_data()
    elif name == "fashion_mnist":
        (xtr, ytr), (xte, yte) = kd.fashion_mnist.load_data()
    elif name == "cifar10":
        (xtr, ytr), (xte, yte) = kd.cifar10.load_data()
    elif name == "kmnist":
        try:
            from extra_keras_datasets import kmnist
        except ImportError as exc:  # pragma: no cover
            raise ImportError("dataset='kmnist' needs the extra-keras-datasets package") from exc
        (xtr, ytr), (xte, yte) = kmnist.load_data()
    else:
        raise ValueError(f"unknown dataset {name!r}; choose one of {DATASETS}")
    return (xtr, ytr.reshape(-1)), (xte, yte.reshape(-1))


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    return np.eye(n_classes)[y]


@dataclass
class Problem:
    """A dataset prepared for one model family, plus a factory for fresh models."""

    kind: str                      # "ann" or "onn"
    dataset: str
    x_train: np.ndarray
    y_train: np.ndarray            # one-hot
    x_test: np.ndarray
    y_test: np.ndarray             # one-hot
    build: Callable[[], tf.keras.Model]

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])


def make_problem(kind: str, dataset: str, *, windowhalfwidth: int = 8,
                 layers: int = 2, hidden: int = 165, bs_error: float = 0.0) -> Problem:
    """
    Prepare data + model factory for a dense ANN (``kind="ann"``) or a mesh ONN
    (``kind="onn"``).

    ANN  - raw images, flattened; one hidden ReLU layer of ``hidden`` units.
           The images are fed as stored (0-255), exactly as in the original runs.
    ONN  - each image is Fourier transformed, low-pass filtered to a
           ``2*windowhalfwidth`` square window, flattened and power-normalised
           (``mesh_onn.helperfunctions.MNISTDataProcessor.fourier``); the model
           is ``layers`` Clements meshes with EO-modulation nonlinearities
           (``mesh_onn.helperfunctions.const_onn_EO``).
    """
    if kind == "ann":
        (xtr, ytr), (xte, yte) = _raw_dataset(dataset)
        input_shape = xtr.shape[1:]

        def build() -> tf.keras.Model:
            model = tf.keras.Sequential([
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=["accuracy"])
            return model

        return Problem("ann", dataset, xtr, one_hot(ytr), xte, one_hot(yte), build)

    if kind == "onn":
        from mesh_onn import helperfunctions as h  # pulls in neurophox

        data = h.MNISTDataProcessor(dataset=dataset).fourier(windowhalfwidth)
        inputsize = int(data.x_train.shape[-1])  # (2*windowhalfwidth)**2 for 28x28 inputs

        def build() -> tf.keras.Model:
            model = h.const_onn_EO(inputsize, L=layers, gamma_pos="out",
                                   bs_error=bs_error, trainlayervectorin=None)
            model.compile(optimizer="adam",
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=["accuracy"])
            return model

        return Problem("onn", dataset, data.x_train, data.y_train, data.x_test, data.y_test, build)

    raise ValueError(f"unknown model kind {kind!r}; choose one of {MODEL_KINDS}")


# --------------------------------------------------------------------------- #
# per-example loss ("hardness") scoring
# --------------------------------------------------------------------------- #
def per_example_loss(model: tf.keras.Model, x: np.ndarray, y: np.ndarray,
                     batch_size: int = 1024) -> np.ndarray:
    """Categorical cross-entropy of every example in ``x`` (same order as ``x``)."""
    preds = model.predict(x, batch_size=batch_size, verbose=0)
    return tf.keras.losses.categorical_crossentropy(y, preds).numpy().astype(np.float64)


def rank_by_loss(losses: np.ndarray, indices: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    ``DataFrame(idx, loss)`` sorted by decreasing loss. ``indices`` maps the
    positions in ``losses`` back to training-set indices (identity by default).
    """
    idx = np.arange(len(losses)) if indices is None else np.asarray(indices)
    order = np.argsort(-losses, kind="stable")
    return pd.DataFrame({"idx": idx[order], "loss": losses[order]})


def hardest(ranking: pd.DataFrame, n: int) -> np.ndarray:
    """Training-set indices of the ``n`` highest-loss examples."""
    return ranking["idx"].to_numpy()[:n].astype(np.int64)


def save_ranking(ranking: pd.DataFrame, path: Path) -> None:
    ranking.to_csv(path, index=False)


def load_ranking(path: Path) -> pd.DataFrame:
    """
    Load a ranking CSV. Accepts both the ``idx,loss`` layout written by these
    scripts and the legacy single-column layout in ``data/losses_per_example``
    (one loss per row, in training-set order).
    """
    df = pd.read_csv(path)
    if {"idx", "loss"}.issubset(df.columns):
        return df.sort_values("loss", ascending=False, kind="stable").reset_index(drop=True)
    losses = df.iloc[:, 0].to_numpy(dtype=np.float64)
    return rank_by_loss(losses)


# --------------------------------------------------------------------------- #
# model / history / config I/O
# --------------------------------------------------------------------------- #
def _keras3() -> bool:
    return str(getattr(tf.keras, "__version__", "2")).startswith("3")


def save_model(model: tf.keras.Model, path: Path) -> Path:
    """SavedModel directory under Keras 2 (as in the original runs), ``.keras`` under Keras 3."""
    path = Path(path)
    if _keras3():
        path = path.with_suffix(".keras")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def load_model(path: Path) -> tf.keras.Model:
    path = Path(path)
    for candidate in (path, path.with_suffix(".keras"), path.with_suffix(".h5")):
        if candidate.exists():
            return tf.keras.models.load_model(str(candidate))
    raise FileNotFoundError(f"no saved model at {path} (.keras/.h5 also tried)")


def save_history(history, path: Path) -> None:
    """Pickle a Keras ``History`` (or its ``.history`` dict), like the original notebooks."""
    hist = history.history if hasattr(history, "history") else dict(history)
    hist = {k: [float(v) for v in vals] for k, vals in hist.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(hist, fh)


def load_history(path: Path) -> Dict[str, list]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_config(cfg: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def problem_from_config(cfg: dict) -> Problem:
    return make_problem(cfg["model"], cfg["dataset"],
                        windowhalfwidth=cfg.get("windowhalfwidth", 8),
                        layers=cfg.get("layers", 2),
                        hidden=cfg.get("hidden", 165),
                        bs_error=cfg.get("bs_error", 0.0))


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_val_accuracy(curves: Dict[str, Sequence[float]], title: str, out_png: Path,
                      dashed: Sequence[str] = (), ylim: Optional[tuple] = None) -> None:
    """
    Overlay validation-accuracy curves (one per entry of ``curves``) and save a PNG.
    Labels listed in ``dashed`` are drawn with a dashed line (used for baselines).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for label, values in curves.items():
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, label=label, linestyle="--" if label in dashed else "-")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_accuracy")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
