from pathlib import Path
from .model import PointNetAE


here = Path(__file__).resolve().parent

AE_BASELINE_NAME = 'final.pt'
AE_BASELINE_WEIGHTS = here / 'weights' / AE_BASELINE_NAME
