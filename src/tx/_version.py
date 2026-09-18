"""Single source of truth for the package version.

Read at build time by ``[tool.hatch.version]`` and bumped by
python-semantic-release; keep it a plain literal assignment.
"""

__version__ = "0.1.0"
