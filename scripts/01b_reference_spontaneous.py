#!/usr/bin/env python
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.simulation.cohort import simulate
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"); p.add_argument("--subject",default="sub-001"); p.add_argument("--duration",type=float,default=100.); a=p.parse_args()
 s=VEPSubject.load(a.data,a.subject); out=ROOT/"outputs/phase1b"; out.mkdir(parents=True,exist_ok=True)
 result=simulate(s,duration=a.duration,noise=True,seed=0); report={"phase":"1B","subject":a.subject,"source":"x2-x1","dt":s.simulator_parameters.dt,"period":s.simulator_parameters.period,"noise":"colored additive ntau=1 seed=0 (not original seed)","finite":True,"duration":a.duration,"diagnostics":result.diagnostics}
 (out/"phase1b_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
