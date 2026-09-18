"""tx — ad-hoc search over Claude Code session transcripts.

The CLI lives in :mod:`tx.cli`; it is imported lazily so that reading
``tx.__version__`` does not pull in orjson.
"""

from tx._version import __version__

__all__ = ["__version__"]
