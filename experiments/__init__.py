"""
Data-pruning / hard-example experiments for mesh ONNs and a dense-ANN control.

Run the scripts as modules from the repository root, e.g.::

    python -m experiments.train_base --model onn --dataset mnist --epochs 5 --out runs/onn_mnist
    python -m experiments.finetune_hardest --run runs/onn_mnist --epochs 20
    python -m experiments.gradual_pruning --run runs/onn_mnist --retention 0.8 --rounds 10
"""
