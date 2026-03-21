"""Vercel serverless function entrypoint.

Vercel deploys this as a single Python function.
The 'includeFiles' glob in vercel.json bundles api.py, modules/, shared/.
For local development, continue using: uvicorn api:app --reload --port 8000
"""

import sys
from pathlib import Path

# The project root is one level up from this file (api/index.py → project root)
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Now standard imports work — api.py, modules/, shared/ are all on the path
from api import app  # noqa: E402
from mangum import Mangum  # noqa: E402

handler = Mangum(app, lifespan="off")