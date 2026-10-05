"""``python -m gmx_harness.checks [DIR]``: list facts files and their staleness (read only)."""

import sys

from .facts import scan

print(scan(sys.argv[1] if len(sys.argv) > 1 else "."))
