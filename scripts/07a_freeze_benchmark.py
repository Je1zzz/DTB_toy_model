#!/usr/bin/env python
import csv,hashlib,json,platform,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort"); OUT=ROOT/"outputs/phase7"; OUT.mkdir(parents=True,exist_ok=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree_sha():
 digest=hashlib.sha256()
 for path in sorted([*ROOT.glob("src/**/*.py"),*ROOT.glob("scripts/*.py"),*ROOT.glob("configs/*")]):
  digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
 return digest.hexdigest()
rows=[]
for subject in sorted(p.name for p in DATA.glob("sub-*")):
 matches=sorted((DATA/subject/"ses-01/ieeg").glob("*task-simulatedseizure_acq-VEPhypothesis_run-01_ieeg.vhdr"))
 if not matches: matches=sorted((DATA/subject/"ses-01/ieeg").glob("*task-simulatedseizure_acq-VEPhypothesis*_ieeg.vhdr"))
 if not matches: continue
 rows.append({"subject_id":subject,"recording_id":matches[0].stem,"seeg_path":str(matches[0]),"electrodes_path":str(DATA/subject/f"{subject}_electrodes.tsv"),"sc_path":str(DATA/"derivatives/tvb"/subject/"struct"/f"{subject}_connectome.zip"),"gain_path":str(DATA/"derivatives/tvb"/subject/"struct"/f"{subject}_gain.txt"),"selection_reason":"run-01" if "run-01" in matches[0].name else "lexicographically-first-valid"})
manifest=OUT/"frozen_manifest.csv"
with manifest.open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
config={"model":"exact fixed-tau reference Stan","n_optimize":50,"n_best":8,"n_chains_per_init":2,"num_warmup":500,"num_samples":500,"adapt_delta":.99,"max_depth":7,"seed":12345,"truth_in_manifest":False}
cfg=OUT/"frozen_config.json"; cfg.write_text(json.dumps(config,indent=2))
yaml=OUT/"frozen_config.yaml"; yaml.write_text("\n".join(f"{key}: {str(value).lower() if isinstance(value,bool) else value}" for key,value in config.items())+"\n")
binary=Path("/data_hdd/hmzhang/vbt_runtime/vbt_stan_prebuilt/5367ef4afb976271ecc6297b70a95dd3d944af61/vep_mcmc")
freeze={"subjects":len(rows),"manifest_sha256":sha(manifest),"config_sha256":sha(cfg),"config_yaml_sha256":sha(yaml),"baseline_git":"not initialized","baseline_code_tree_sha256":tree_sha(),"reference_repo_sha":"5367ef4afb976271ecc6297b70a95dd3d944af61","vep_generator_sha":"a99c88354015f4c961d49a92c076ed0b675c740b","stan_binary":str(binary),"stan_binary_sha256":sha(binary),"python":platform.python_version(),"numpy":np.__version__}
(OUT/"freeze_manifest.json").write_text(json.dumps(freeze,indent=2)); print(json.dumps(freeze,indent=2))
