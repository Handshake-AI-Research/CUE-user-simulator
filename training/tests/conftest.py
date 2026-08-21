"""Test-session setup that must happen before torch or scikit-learn are imported."""

from __future__ import annotations

import os

# torch and scikit-learn each bring their own OpenMP runtime. Once both are loaded into
# one process, sklearn's parallel regions (e.g. PCA / KMeans) can deadlock on entering a
# contended OpenMP region, which hangs the whole suite -- the tests pass fine when the
# file runs alone, so this only shows up in a full run. Pinning OpenMP to one thread
# avoids the contended region entirely; the suite is CPU-trivial, so nothing gets slower.
os.environ.setdefault("OMP_NUM_THREADS", "1")
