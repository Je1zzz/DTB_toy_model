#!/usr/bin/env python
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.simulation.cohort import simulate
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"); p.add_argument("--duration",type=float,default=4500); a=p.parse_args(); s=VEPSubject.load(a.data,"sub-001"); out=ROOT/"outputs/phase1b"; out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for label,noise,dt in [("deterministic",False,s.simulator_parameters.dt),("reference_noise",True,s.simulator_parameters.dt),("half_dt_diagnostic",True,s.simulator_parameters.dt/2)]:
  try: r=simulate(s,duration=a.duration,noise=noise,seed=0,dt=dt); rows.append({"case":label,"status":"PASS","dt":dt,"last":r.diagnostics[-1] if r.diagnostics else {}})
  except Exception as e: rows.append({"case":label,"status":"FAIL","dt":dt,"error":f"{type(e).__name__}: {e}"})
 report={"phase":"1B-long","duration":a.duration,"no_clipping":True,"cases":rows,"G1B_LONG":"PASS" if rows[1]["status"]=="PASS" else "FAIL","mechanistic_replay":"NOT FULLY VERIFIED","reason":"generation-era TVB version and original RNG state unavailable; colored-noise compatibility is not point-wise reference-verified","old_overflow_root_cause":"ROOT CAUSE NOT PROVEN; reference-semantics mismatch is demonstrated"}; (out/"long_window_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
