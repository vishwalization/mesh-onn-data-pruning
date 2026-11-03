# Per-example loss ("hardness") scores

Categorical cross-entropy of every **training** example under a briefly
trained 2-layer Clements-mesh ONN fed with low-pass Fourier features
(`mesh_onn.helperfunctions.MNISTDataProcessor.fourier`; the scoring step is
what `experiments/train_base.py` does after training). These are the scores
that rank examples for the hard-subset experiments. Each file has one column (`0`) with one loss per row,
in the dataset's original training order, so row *i* is the loss of training
image *i*.

| File | Dataset | Rows |
|---|---|---|
| `mnist_onn_loss_per_example.csv` | MNIST | 60 000 |
| `fashion_mnist_onn_loss_per_example_run1.csv` | Fashion-MNIST | 60 000 |
| `fashion_mnist_onn_loss_per_example_run2.csv` | Fashion-MNIST (independent run) | 60 000 |
| `cifar10_onn_loss_per_example.csv` | CIFAR-10 | 50 000 |

`experiments.common.load_ranking` reads this legacy single-column layout as well
as the `idx,loss` rankings written by the scripts, so any of these files can be
passed straight to `finetune_hardest.py --ranking <file>`:

```python
from experiments import common as C
ranking = C.load_ranking("data/losses_per_example/mnist_onn_loss_per_example.csv")
hardest_10k = C.hardest(ranking, 10_000)   # training-set indices, hardest first
```
