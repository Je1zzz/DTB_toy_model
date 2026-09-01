#!/usr/bin/env python
"""Truth-free runner: this file intentionally only reads the frozen manifest."""
import argparse,csv,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/phase7"; ENGINE=ROOT/"src/vbt/inference/reference_engine.py"
def one(row,full,run_root):
 subject=row["subject_id"]; target=run_root/"inference"/subject; target.mkdir(parents=True,exist_ok=True)
 if (target/"blind_run_report.json").exists(): return {"subject":subject,"status":"SKIPPED_COMPLETE"}
 if any(target.iterdir()): return {"subject":subject,"status":"FAIL_FORBIDDEN_OLD_ARTIFACTS","return_code":99}
 params=["--opt-starts","50","--best-inits","8","--chains","16","--opt-iter","2000","--warmup","500","--samples","500"] if full else ["--opt-starts","2","--best-inits","1","--chains","2","--opt-iter","50","--warmup","5","--samples","5"]
 cmd=[sys.executable,str(ENGINE),"--blind-only","--subject",subject,"--output",str(target),"--vhdr",row["seeg_path"],"--electrodes",row["electrodes_path"],"--sc",row["sc_path"],"--gain",row["gain_path"],*params,"--seed",str(12345+int(subject[-3:])*1000)]; env=os.environ.copy(); env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}); started=time.time()
 with (target/"batch.log").open("w") as h: rc=subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,env=env).returncode
 return {"subject":subject,"status":"PASS" if rc==0 else "FAIL","return_code":rc,"seconds":time.time()-started}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--jobs",type=int,default=2); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
 with (OUT/"frozen_manifest.csv").open() as h: rows=list(csv.DictReader(h))
 results=[]
 with ThreadPoolExecutor(max_workers=a.jobs) as pool:
  run_root=OUT if not a.smoke else ROOT/"outputs/phase7_smoke"
  run_root.mkdir(parents=True,exist_ok=True)
  (run_root/"run_mode.json").write_text(json.dumps({"mode":"FULL_PRIMARY" if not a.smoke else "ENGINEERING_SMOKE_ONLY","frozen_primary_config_executed":not a.smoke},indent=2))
  jobs={pool.submit(one,row,not a.smoke,run_root):row for row in rows}
  for f in as_completed(jobs): results.append(f.result()); print(json.dumps(results[-1]),flush=True)
 (run_root/"blind_run_status.json").write_text(json.dumps(sorted(results,key=lambda x:x["subject"]),indent=2))
 hashes={p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run_root/"inference").glob("sub-*/prediction_blind.csv"))}
 (run_root/"prediction_hashes_pre_unlock.json").write_text(json.dumps(hashes,indent=2))
if __name__=="__main__": main()
