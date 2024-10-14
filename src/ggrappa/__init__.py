"""gGRAPPA.

gGRAPPA provides an easy to use generalized GRAPPA reconstruction tool for kspace data.
"""
import torch

from dataclasses import dataclass
from typing import List


@dataclass
class GRAPPAReconSpec:
    weights: torch.Tensor
    sbl: List[int]
    tbl: List[int]
    pos: List[int]
    af: List[int]
    delta: int


from .grappaND import GRAPPA_Recon

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass