"""
mesh_onn
========

Simulation code for *maximally error-tolerant* Mach-Zehnder-interferometer (MZI)
mesh optical neural networks (ONNs), built on top of ``neurophox``.

The modules in this package are derived from the MIT Quantum Photonics Group
repository ``QPG-MIT/Maximally-Error-Tolerant-Mesh-ONNs`` (MIT License) and
were adapted during a data-pruning study of mesh ONNs:

* :mod:`mesh_onn.helperfunctions`  – data pipeline (``MNISTDataProcessor``),
  the EO intensity-modulation nonlinearity, ``const_onn_EO`` (builds a
  Clements-mesh ONN as a Keras ``Sequential``), phase extraction and the
  Bandyopadhyay-style hardware error-correction routines.
* :mod:`mesh_onn.mainfunctions`    – training drivers: direct training of faulty
  meshes, "adiabatic" transfer training across beamsplitter error levels,
  lossy-mesh evaluation and result aggregation.
* :mod:`mesh_onn.errcorr3mzi_riemann` – Clements decomposition with
  beamsplitter errors (2-MZI / 3-MZI) accelerated with ``numba``.
* :mod:`mesh_onn.allcalls`         – the upstream driver script (top-level
  code; run with ``python -m mesh_onn.allcalls``).

Importing the sub-modules pulls in TensorFlow and neurophox, so nothing is
imported eagerly here.
"""

__all__ = ["helperfunctions", "mainfunctions", "errcorr3mzi_riemann", "allcalls"]
