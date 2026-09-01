#!/usr/bin/env python
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.inference.reference_engine import main
if __name__=="__main__": sys.argv.extend(["--prepare-only","--subject","sub-001","--output",str(ROOT/"outputs/phase5/sub-001/input")]); raise SystemExit(main())
