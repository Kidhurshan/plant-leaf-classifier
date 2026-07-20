"""EgyPLI plant-leaf species classification -- all project logic.

Notebooks and scripts are thin drivers that import from this package. Keeping
every model definition, data pipeline and training loop here guarantees the
three models are trained by *identical* shared code, so the only variable in the
comparison is the backbone.
"""

__version__ = "1.0.0"
