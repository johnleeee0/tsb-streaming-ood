"""Main OOD detector set (17 detectors).

Importing this package imports every detector submodule so that each detector's
``@register_ood(...)`` decorator fires and the class is discoverable via
``core.registry.OOD_REGISTRY``.
"""

from __future__ import annotations

# Post-hoc / logit-based
from . import msp
from . import odin
from . import energy
from . import mahalanobis
from . import dfm_pca
from . import react
from . import dice
from . import scale
from . import gradnorm
# Feature / reconstruction / distance based
from . import srs
from . import dimmad
from . import catsight
from . import codit
from . import invad
from . import m2n2
from . import deedee
from . import diffad

__all__ = [
    "msp", "odin", "energy", "mahalanobis", "dfm_pca", "react", "dice", "scale",
    "gradnorm", "srs", "dimmad", "catsight", "codit", "invad", "m2n2", "deedee",
    "diffad",
]
