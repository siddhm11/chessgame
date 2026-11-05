"""pytest bootstrap: put the project root on sys.path and make sure the
synthetic fixtures exist before any test runs."""

import os
import sys
from pathlib import Path

# Tests that don't pick a backend explicitly use the deterministic template
# matcher (as CI does); tests of the model build ModelBackend themselves.
os.environ.setdefault("CVC_BACKEND", "classical")

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.generate_fixtures import generate  # noqa: E402

generate(force=False)
