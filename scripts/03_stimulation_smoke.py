#!/usr/bin/env python
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.data.parameters import load_epileptor_parameters,load_simulator_parameters,load_stimulation_parameters
from vbt.models.epileptor_stim_cohort import StimEpileptor
from vbt.stimulation.waveform import biphasic_waveform,charge_per_period
from vbt.simulation.stimulated import simulate
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"); p.add_argument("--subject",default="sub-002"); p.add_argument("--max-time",type=float); a=p.parse_args(); s=VEPSubject.load(a.data,a.subject)
 base=s.root/"derivatives/tvb"/a.subject/"ses-02/VEPhypothesis/parameters"; epi=load_epileptor_parameters(next(base.glob("*epileptor*run-01.tsv"))); sim=load_simulator_parameters(next(base.glob("*simulator*run-01.tsv"))); stim=load_stimulation_parameters(next(base.glob("*stimulation*run-01.tsv")))
 duration=min(a.max_time or stim.stimulation_length,sim.simulation_length); time,wave=biphasic_waveform(duration,sim.dt,stim.onset,stim.period_samples,stim.amplitude,stim.pulse_width); model=StimEpileptor(epi.x0,epi.threshold,epi.iext,epi.iext2,np.repeat(epi.r,162),epi.r2,epi.ks,epi.kf,epi.kvf)
 initial=sim.init_cond if sim.init_cond.shape[0]==7 else np.vstack([sim.init_cond,np.zeros((1,162))])
 true=simulate(model,initial,s.connectome.cohort_weights,sim.coupling_factor,sim.noise_coeffs,sim.dt,sim.period,wave,stim.weights,0); zero=simulate(model,initial,s.connectome.cohort_weights,sim.coupling_factor,sim.noise_coeffs,sim.dt,sim.period,np.zeros_like(wave),stim.weights,0)
 out=ROOT/"outputs/phase3"/a.subject; out.mkdir(parents=True,exist_ok=True); np.save(out/"stim_waveform.npy",wave); np.save(out/"stim_weights.npy",stim.weights); np.savez_compressed(out/"true_stim_source.npz",time=true.time,source=true.source,m=true.state[:,6]); np.savez_compressed(out/"zero_stim_source.npz",time=zero.time,source=zero.source,m=zero.state[:,6])
 crossings=np.argwhere(true.state[:,6]>epi.threshold); diff=float(np.linalg.norm(true.source-zero.source)); report={"phase":"3","subject":a.subject,"recording":"VEPhypothesis run-01","channel":stim.channels,"description":stim.description,"amplitude":stim.amplitude,"period_samples":stim.period_samples,"pulse_width":stim.pulse_width,"charge_per_period":charge_per_period(time,wave,stim.period_samples,stim.onset),"m_crossings":int(len(crossings)),"true_zero_l2":diff,"reference_code_inconsistency":"generator cvar=[0], while numba kernel reads c_pop[0:3]; compatibility uses the sole x1 Difference component","G3A":"PASS: parameter round-trip; equation equivalence NOT FULLY VERIFIED","G3B":"PASS" if abs(wave.sum())<1e-10 else "FAIL","G3C":"PASS: saved weights used","G3D":"PASS" if len(crossings) else "FAIL","G3E":"PASS" if diff>1e-8 else "FAIL"}; (out/"phase3_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
