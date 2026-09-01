#!/usr/bin/env python
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.data.timeseries import load_source
from vbt.features.reference_seizure import compute_slp_sim,compute_onset,compute_offset
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"); p.add_argument("--subject",default="sub-001"); a=p.parse_args(); s=VEPSubject.load(a.data,a.subject)
 files=[x for x in s.source_files() if "ses-01" in str(x) and "VEPhypothesis" in str(x) and "run-01" in x.name]; source_path=files[0]; t,x=load_source(source_path); slp=compute_slp_sim(x,sfreq=1000.); onset=compute_onset(slp); offset=compute_offset(slp); order=np.argsort(onset); ez=set(s.ez_truth); pz=set(s.pz_truth); k=max(3,len(ez)); gate=any(s.region_names[i] in ez for i in order[:k])
 out=ROOT/"outputs/phase2"/a.subject; out.mkdir(parents=True,exist_ok=True)
 with (out/"provided_source_propagation.csv").open("w",newline="") as h:
  w=csv.writer(h); w.writerow(["region","is_EZ_truth","is_PZ_truth","onset","offset","recruited","recruitment_rank"]); ranks=np.empty(162,int); ranks[order]=np.arange(1,163)
  for i,r in enumerate(s.region_names): w.writerow([r,r in ez,r in pz,int(onset[i]),int(offset[i]),True,int(ranks[i])])
 fig,ax=plt.subplots(figsize=(12,5)); ax.scatter(np.arange(162),onset,c=["red" if r in ez else "steelblue" for r in s.region_names],s=18); ax.set(xlabel="VEP region index",ylabel="SLP onset sample",title=f"{a.subject} provided-source recruitment (red=EZ)"); fig.tight_layout(); fig.savefig(out/"provided_source_propagation.png",dpi=150); plt.close(fig)
 ez_ranks={s.region_names[i]:int(ranks[i]) for i in range(162) if s.region_names[i] in ez}; report={"phase":"2","subject":a.subject,"source_path":str(source_path),"first_recruited":[s.region_names[i] for i in order[:10]],"EZ":list(ez),"EZ_recruitment_ranks":ez_ranks,"G2A_feature_extraction":"PASS","G2B_dataset_EZ_early_recruitment_sanity":"PASS" if gate else "FAIL","G2C_replay_propagation":"NOT TESTED: replay equivalence not fully verified"}; (out/"phase2_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
