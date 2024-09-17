"""gGRAPPA.

gGRAPPA provides an easy to use generalized GRAPPA reconstruction tool for kspace data.
"""

from .grappaND import GRAPPA_Recon

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass