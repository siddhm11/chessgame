"""pytest bootstrap: put the project root on sys.path and make sure the
synthetic fixtures exist before any test runs."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.generate_fixtures import generate  # noqa: E402

generate(force=False)
