"""Thin shared test setup — no app server, no DB, no network.

Puts the repo root on ``sys.path`` so ``eval_harness`` and the ``examples``
modules resolve even when tests run without an editable install.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
