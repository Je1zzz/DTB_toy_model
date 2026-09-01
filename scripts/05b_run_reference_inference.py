#!/usr/bin/env python
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.inference.reference_engine import main
if __name__=="__main__":
 # Engineering smoke only; Phase 7 freezes the full 50/8x2/500/500 configuration.
 sys.argv.extend(["--blind-only","--subject","sub-001","--output",str(ROOT/"outputs/phase5/sub-001"),"--opt-starts","8","--best-inits","2","--chains","4","--opt-iter","200","--warmup","50","--samples","50"]); raise SystemExit(main())
