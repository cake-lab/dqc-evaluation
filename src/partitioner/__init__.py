from src.partitioner.base import Partitioner
from src.partitioner.fgp_roee import FGPrOEEPartitioner
from src.partitioner.mlfm_r import MLFMRPartitioner
from src.partitioner.pytket_dqc import (
    PytketDqcESDPartitioner,
    PytketDqcPEPartitioner,
    PytketDqcPPartitioner,
)

__all__ = [
    "FGPrOEEPartitioner",
    "MLFMRPartitioner",
    "Partitioner",
    "PytketDqcESDPartitioner",
    "PytketDqcPEPartitioner",
    "PytketDqcPPartitioner",
]

PARTITIONERS = {
    "mlfm-r": MLFMRPartitioner,
    "fgp-roee": FGPrOEEPartitioner,
    "pytket-dqc-p": PytketDqcPPartitioner,
    "pytket-dqc-pe": PytketDqcPEPartitioner,
    "pytket-dqc-esd": PytketDqcESDPartitioner,
}
