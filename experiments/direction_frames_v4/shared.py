"""Reuse versioned utilities; v4 has no navigation-model dependencies."""
from pathlib import Path
import sys
BASE4=Path(__file__).resolve().parent
V3=BASE4.parent/'direction_frames_v3'
sys.path.insert(0,str(V3))
from common import *
OUT=BASE4/'results'
DATA=Path('/home/20261024_1/Workspace/DATA/sbfnav-data')
CUE=re.compile(r'\b(?:facing|looking|coming\s+from|approaching|along|viewed|viewing|towards?|stand(?:ing)?)\b',re.I)
# Restore v4-first lookup after legacy common adds its own v2 utility path.
sys.path.insert(0,str(BASE4))
sys.path.insert(1,str(V3))
